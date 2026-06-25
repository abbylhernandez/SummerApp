/**
 * EMG signal simulator for STM32 Nucleo-L476RG
 *
 * Drives PB13 (TIM1_CH1N) with PWM. After an RC low-pass filter, the voltage
 * mimics a single EMG channel. Split that one analog line to all three ADC
 * inputs on the receiving MCU — every channel will read the same value.
 *
 * Wiring:
 *   PB13 -> [optional 220-470 ohm] -> RC filter (e.g. 1 kOhm + 100 nF to GND)
 *        -> tap the filtered node to EMG ch1, ch2, and ch3 on the other board
 *   GND  -> common ground between both boards
 *
 * Note: PB13 also drives the green user LED on the Nucleo. For a cleaner
 * analog signal, remove jumper SB8 (disconnects the LED) or ignore the LED.
 */

#include <stdint.h>

/* 64-point sine lookup (one cycle), values scaled x10000 */
static const int16_t g_sine_lut[64] = {
    0, 980, 1951, 2903, 3827, 4714, 5556, 6344,
    7071, 7730, 8315, 8819, 9239, 9568, 9802, 9939,
    9976, 9913, 9750, 9490, 9135, 8688, 8155, 7540,
    6850, 6092, 5274, 4404, 3492, 2548, 1582, 608,
    -608, -1582, -2548, -3492, -4404, -5274, -6092, -6850,
    -7540, -8155, -8688, -9135, -9490, -9750, -9913, -9976,
    -9939, -9802, -9568, -9239, -8819, -8315, -7730, -7071,
    -6344, -5556, -4714, -3827, -2903, -1951, -980, 0
};

/* ---- CMSIS-style register map (minimal) ---- */
#define RCC_BASE            (0x40021000UL)
#define GPIOB_BASE          (0x48000400UL)
#define TIM1_BASE           (0x40012C00UL)
#define SysTick_BASE        (0xE000E010UL)

#define RCC                 ((RCC_TypeDef *)RCC_BASE)
#define GPIOB               ((GPIO_TypeDef *)GPIOB_BASE)
#define TIM1                ((TIM_TypeDef *)TIM1_BASE)

typedef struct {
    volatile uint32_t CTRL;
    volatile uint32_t LOAD;
    volatile uint32_t VAL;
    volatile uint32_t CALIB;
} SysTick_TypeDef;

#define SysTick             ((SysTick_TypeDef *)SysTick_BASE)

typedef struct {
    volatile uint32_t CR;
    volatile uint32_t ICSCR;
    volatile uint32_t CFGR;
    volatile uint32_t PLLCFGR;
    volatile uint32_t RESERVED0;
    volatile uint32_t CIER;
    volatile uint32_t CIFR;
    volatile uint32_t CICR;
    volatile uint32_t AHB1ENR;
    volatile uint32_t AHB2ENR;
    volatile uint32_t AHB3ENR;
    volatile uint32_t RESERVED1;
    volatile uint32_t APB1ENR1;
    volatile uint32_t APB1ENR2;
    volatile uint32_t APB2ENR;
} RCC_TypeDef;

typedef struct {
    volatile uint32_t MODER;
    volatile uint32_t OTYPER;
    volatile uint32_t OSPEEDR;
    volatile uint32_t PUPDR;
    volatile uint32_t IDR;
    volatile uint32_t ODR;
    volatile uint32_t BSRR;
    volatile uint32_t LCKR;
    volatile uint32_t AFR[2];
} GPIO_TypeDef;

typedef struct {
    volatile uint32_t CR1;
    volatile uint32_t CR2;
    volatile uint32_t SMCR;
    volatile uint32_t DIER;
    volatile uint32_t SR;
    volatile uint32_t EGR;
    volatile uint32_t CCMR1;
    volatile uint32_t CCMR2;
    volatile uint32_t CCER;
    volatile uint32_t CNT;
    volatile uint32_t PSC;
    volatile uint32_t ARR;
    volatile uint32_t RCR;
    volatile uint32_t CCR1;
    volatile uint32_t CCR2;
    volatile uint32_t CCR3;
    volatile uint32_t CCR4;
    volatile uint32_t BDTR;
    volatile uint32_t DCR;
    volatile uint32_t DMAR;
    volatile uint32_t OR;
} TIM_TypeDef;

#define RCC_AHB2ENR_GPIOBEN   (1U << 1)
#define RCC_APB2ENR_TIM1EN    (1U << 11)

#define TIM_CR1_CEN           (1U << 0)
#define TIM_CCER_CC1E         (1U << 0)
#define TIM_CCER_CC1NE        (1U << 2)
#define TIM_BDTR_MOE          (1U << 15)

/* Matches config.py on the Python logger side */
#define VDD_V                 3.3f
#define EMG_VREF              3.0f
#define ADC_MAX_COUNTS        4095.0f

/* PWM: 40 kHz carrier is easy to filter with a small RC network */
#define PWM_ARR               99U
#define SAMPLE_HZ             1000U

/* SysTick reload for 1 ms ticks at 4 MHz MSI (default out of reset) */
#define SYSCLK_HZ             4000000UL
#define SYSTICK_RELOAD        ((SYSCLK_HZ / SAMPLE_HZ) - 1U)

static volatile uint32_t g_tick_ms = 0U;

static float clampf(float x, float lo, float hi)
{
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

/* 64-point sine lookup (one cycle), output in [-1, 1] — no libm needed */
static float fast_sin(float cycles)
{
    int32_t idx = (int32_t)(cycles * 64.0f);
    idx &= 63;
    return (float)g_sine_lut[idx] / 10000.0f;
}

static float fast_fabsf(float x)
{
    return (x < 0.0f) ? -x : x;
}

/* EMG-like waveform: burst envelope + high-frequency ripple + slow drift */
static float fake_emg_volts(uint32_t t_ms)
{
    const float t = (float)t_ms * 0.001f;

    /* ~1.2 Hz contraction bursts (0 = rest, 1 = active) */
    const float burst_raw = fast_sin(2f * t);
    const float burst = burst_raw > 0.15f ? (burst_raw - 0.15f) / 0.85f : 0.0f;

    const float ripple = fast_fabsf(fast_sin(90.0f * t));

    const float wander = 0.1f * fast_sin(0.35f * t + 0.4f);

    /* Secondary ripple during bursts */
    const float texture = 1.00f * fast_fabsf(fast_sin(140.0f * t + 1.1f));

    /* Center ~1.4 V at rest; bursts push toward ~2.8-3.0 V */
    const float volts = 1.40f + wander + burst * (0.70f + 1.85f * ripple + texture);

    return clampf(volts, 0.90f, 2.95f);
}

static uint32_t volts_to_ccr(float volts)
{
    const float duty = clampf(volts / VDD_V, 0.0f, 1.0f);

    /* TIM1_CH1N is the complementary output (inverted vs CH1) */
    const float inv_duty = 1.0f - duty;
    return (uint32_t)(inv_duty * ((float)PWM_ARR + 1.0f));
}

static void clock_enable_peripherals(void)
{
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOBEN;
    RCC->APB2ENR |= RCC_APB2ENR_TIM1EN;
}

static void gpio_pb13_tim1_ch1n(void)
{
    /* PB13 = alternate function 1 = TIM1_CH1N */
    GPIOB->MODER = (GPIOB->MODER & ~(3U << 26)) | (2U << 26);
    GPIOB->AFR[1] = (GPIOB->AFR[1] & ~(0xFU << 20)) | (1U << 20);
    GPIOB->OSPEEDR |= (3U << 26);
}

static void tim1_pwm_init(void)
{
    TIM1->CR1 = 0U;
    TIM1->PSC = 0U;
    TIM1->ARR = PWM_ARR;
    TIM1->CCR1 = PWM_ARR; /* start near 0 V after filtering */
    TIM1->RCR = 0U;

    /* OC1M = 110: PWM mode 1 */
    TIM1->CCMR1 = (6U << 4);
    TIM1->CCER = TIM_CCER_CC1E | TIM_CCER_CC1NE;
    TIM1->BDTR = TIM_BDTR_MOE;
    TIM1->EGR = (1U << 0);
    TIM1->CR1 = TIM_CR1_CEN;
}

static void systick_init(void)
{
    g_tick_ms = 0U;
    SysTick->LOAD = SYSTICK_RELOAD;
    SysTick->VAL = 0U;
    /* enable | tickint | processor clock */
    SysTick->CTRL = (1U << 2) | (1U << 1) | (1U << 0);
}

void SysTick_Handler(void)
{
    g_tick_ms++;
}

int main(void)
{
    clock_enable_peripherals();
    gpio_pb13_tim1_ch1n();
    tim1_pwm_init();
    systick_init();

    for (;;) {
        const uint32_t tick = g_tick_ms;
        const float volts = fake_emg_volts(tick);
        TIM1->CCR1 = volts_to_ccr(volts);

        while (g_tick_ms == tick) {
            __asm volatile("wfi");
        }
    }
}
