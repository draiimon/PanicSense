animation: {
    "accordion-down": "accordion-down 0.2s ease-out",
    "accordion-up": "accordion-up 0.2s ease-out",
    "ripple-slow": "ripple 3s linear infinite",
  },
  keyframes: {
    ripple: {
      "0%": { transform: "scale(0.7)", opacity: "0.5" },
      "100%": { transform: "scale(2)", opacity: "0" },
    },
  },
  utilities: {
    '.animation-delay-1000': {
      'animation-delay': '1000ms',
    },
  },