import { motion } from 'framer-motion'
import clsx from 'clsx'

/**
 * LUXURY MEDICAL AI - Premium Card Component System
 * Variants: glass, elevated, glow-primary, glow-accent, ghost
 */
export function Card({
  children,
  variant = 'glass',
  className = '',
  hover = true,
  animate = true,
  padding = 'default',
  ...props
}) {
  const baseClasses = 'relative overflow-hidden transition-all duration-400'

  const paddingClasses = {
    none: '',
    sm: 'p-4',
    default: 'p-6',
    lg: 'p-8',
    xl: 'p-10',
  }

  const variants = {
    glass: 'card-glass',
    elevated: 'card-elevated',
    'glow-primary': 'card-glow-primary',
    'glow-accent': 'card-glow-accent',
    ghost: 'bg-transparent border border-transparent hover:bg-white/5',
    ceramic: 'card-elevated', // legacy support
  }

  const classes = clsx(
    baseClasses,
    variants[variant],
    paddingClasses[padding] || paddingClasses.default,
    hover && 'cursor-pointer',
    className
  )

  const Component = animate ? motion.div : 'div'

  const motionProps = animate ? {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    whileHover: hover ? { y: -4, scale: 1.01 } : {},
    transition: { duration: 0.4, ease: [0.4, 0, 0.2, 1] }
  } : {}

  return (
    <Component className={classes} {...motionProps} {...props}>
      {children}
    </Component>
  )
}

/**
 * Card Header - For titles and actions
 */
export function CardHeader({ children, className = '' }) {
  return (
    <div className={clsx('mb-4 flex items-center justify-between', className)}>
      {children}
    </div>
  )
}

/**
 * Card Title - Premium typography
 */
export function CardTitle({ children, className = '', gradient = false }) {
  return (
    <h3 className={clsx(
      'text-lg font-semibold tracking-tight',
      gradient ? 'text-gradient-primary' : 'text-text-primary',
      className
    )}>
      {children}
    </h3>
  )
}

/**
 * Card Description - Secondary text
 */
export function CardDescription({ children, className = '' }) {
  return (
    <p className={clsx('text-sm text-text-secondary leading-relaxed', className)}>
      {children}
    </p>
  )
}

/**
 * Card Data - Monospace for numbers/metrics with optional glow
 */
export function CardData({ children, label, className = '', glow = false, size = 'default' }) {
  const sizes = {
    sm: 'text-xl',
    default: 'text-2xl',
    lg: 'text-4xl',
    xl: 'text-6xl',
  }

  return (
    <div className={clsx('space-y-1', className)}>
      {label && (
        <div className="text-xs uppercase tracking-wider text-text-tertiary font-medium">
          {label}
        </div>
      )}
      <div className={clsx(
        'font-mono font-bold tracking-tight',
        sizes[size],
        glow ? 'text-gradient-primary' : 'text-text-primary'
      )}>
        {children}
      </div>
    </div>
  )
}

/**
 * Card Badge - Status indicators
 */
export function CardBadge({ children, variant = 'default', className = '' }) {
  const variants = {
    default: 'bg-primary-500/10 text-primary-500 border-primary-500/20',
    accent: 'bg-accent-500/10 text-accent-400 border-accent-500/20',
    ai: 'bg-ai-500/10 text-ai-400 border-ai-500/20',
    success: 'bg-success/10 text-success border-success/20',
    warning: 'bg-warning/10 text-warning border-warning/20',
    danger: 'bg-danger/10 text-danger border-danger/20',
  }

  return (
    <span className={clsx(
      'inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider border',
      variants[variant],
      className
    )}>
      {children}
    </span>
  )
}

/**
 * IconWrapper - Circular icon containers with glow
 */
export function IconWrapper({ children, variant = 'primary', size = 'default', className = '' }) {
  const sizes = {
    sm: 'w-10 h-10',
    default: 'w-14 h-14',
    lg: 'w-16 h-16',
  }

  const variants = {
    primary: 'bg-primary-500/10 text-primary-500',
    accent: 'bg-accent-500/10 text-accent-400',
    ai: 'bg-ai-500/10 text-ai-400',
  }

  return (
    <div className={clsx(
      'rounded-2xl flex items-center justify-center transition-transform duration-300 group-hover:scale-110',
      sizes[size],
      variants[variant],
      className
    )}>
      {children}
    </div>
  )
}
