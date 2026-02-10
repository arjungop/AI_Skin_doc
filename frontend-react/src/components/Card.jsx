import { motion } from 'framer-motion'
import clsx from 'clsx'

/**
 * MODERN CONSUMER HEALTH CARD SYSTEM
 * Simple, Clean, White cards with soft shadows.
 */
export function Card({
  children,
  className = '',
  hover = true,
  animate = true,
  padding = 'default',
  ...props
}) {
  const baseClasses = 'bg-white rounded-2xl border border-slate-100 shadow-soft-md transition-all duration-300'

  const paddingClasses = {
    none: '',
    sm: 'p-4',
    default: 'p-6',
    lg: 'p-8',
    xl: 'p-10',
  }

  const hoverClasses = hover ? 'hover:shadow-soft-xl hover:-translate-y-1 cursor-pointer' : ''

  const classes = clsx(
    baseClasses,
    paddingClasses[padding],
    hoverClasses,
    className
  )

  const Component = animate ? motion.div : 'div'

  const motionProps = animate ? {
    initial: { opacity: 0, y: 10 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }
  } : {}

  return (
    <Component className={classes} {...motionProps} {...props}>
      {children}
    </Component>
  )
}

export function CardHeader({ children, className = '' }) {
  return (
    <div className={clsx('mb-4 flex items-center justify-between', className)}>
      {children}
    </div>
  )
}

export function CardTitle({ children, className = '' }) {
  return (
    <h3 className={clsx('text-lg font-bold text-slate-900 tracking-tight', className)}>
      {children}
    </h3>
  )
}

export function CardDescription({ children, className = '' }) {
  return (
    <p className={clsx('text-sm text-slate-500 leading-relaxed', className)}>
      {children}
    </p>
  )
}

export function CardData({ children, label, className = '', size = 'default' }) {
  const sizes = {
    sm: 'text-xl',
    default: 'text-2xl',
    lg: 'text-4xl',
    xl: 'text-5xl',
  }

  return (
    <div className={clsx('space-y-1', className)}>
      {label && (
        <div className="text-xs uppercase tracking-wider text-slate-400 font-semibold">
          {label}
        </div>
      )}
      <div className={clsx(
        'font-display font-bold text-slate-900',
        sizes[size]
      )}>
        {children}
      </div>
    </div>
  )
}

export function CardBadge({ children, variant = 'default', className = '' }) {
  const variants = {
    default: 'bg-slate-100 text-slate-600',
    primary: 'bg-primary-50 text-primary-700',
    success: 'bg-emerald-50 text-emerald-700',
    warning: 'bg-amber-50 text-amber-700',
    danger: 'bg-rose-50 text-rose-700',
    ai: 'bg-violet-50 text-violet-700',
  }

  return (
    <span className={clsx(
      'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wide',
      variants[variant],
      className
    )}>
      {children}
    </span>
  )
}

export function IconWrapper({ children, variant = 'default', size = 'default', className = '' }) {
  const sizes = {
    sm: 'w-10 h-10',
    default: 'w-12 h-12',
    lg: 'w-16 h-16',
  }

  const variants = {
    default: 'bg-slate-100 text-slate-600',
    primary: 'bg-primary-50 text-primary-600',
    success: 'bg-emerald-50 text-emerald-600',
    ai: 'bg-violet-50 text-violet-600',
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
