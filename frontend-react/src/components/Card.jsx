import { motion } from 'framer-motion'
import clsx from 'clsx'

/**
 * CLINICAL FUTURISM - Unified Card Component
 * Variants: ceramic, glass, ghost
 */
export function Card({ 
  children, 
  variant = 'ceramic', 
  className = '',
  hover = true,
  animate = true,
  ...props 
}) {
  const baseClasses = 'p-6 transition-all duration-300'
  
  const variants = {
    ceramic: 'card-ceramic',
    glass: 'card-glass',
    ghost: 'card-ghost',
  }
  
  const classes = clsx(
    baseClasses,
    variants[variant],
    hover && 'cursor-pointer',
    className
  )
  
  const Component = animate ? motion.div : 'div'
  
  const motionProps = animate ? {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    whileHover: hover ? { y: -4 } : {},
    transition: { duration: 0.3 }
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
 * Card Title - Tight, technical typography
 */
export function CardTitle({ children, className = '' }) {
  return (
    <h3 className={clsx('text-lg font-semibold text-text-primary tracking-tighter', className)}>
      {children}
    </h3>
  )
}

/**
 * Card Description - Secondary text
 */
export function CardDescription({ children, className = '' }) {
  return (
    <p className={clsx('text-sm text-text-secondary', className)}>
      {children}
    </p>
  )
}

/**
 * Card Data - Monospace for numbers/metrics
 */
export function CardData({ children, label, className = '' }) {
  return (
    <div className={clsx('space-y-1', className)}>
      {label && (
        <div className="text-xs uppercase tracking-wider text-text-secondary font-medium">
          {label}
        </div>
      )}
      <div className="font-mono text-2xl font-semibold text-text-mono tracking-tight">
        {children}
      </div>
    </div>
  )
}
