import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

export default function RiskMeter({ percentage, riskLevel }) {
  const [displayPercentage, setDisplayPercentage] = useState(0)

  useEffect(() => {
    // Animate the number counting up
    const duration = 1500
    const steps = 60
    const stepDuration = duration / steps
    const increment = percentage / steps
    let current = 0

    const timer = setInterval(() => {
      current += increment
      if (current >= percentage) {
        setDisplayPercentage(percentage)
        clearInterval(timer)
      } else {
        setDisplayPercentage(Math.round(current))
      }
    }, stepDuration)

    return () => clearInterval(timer)
  }, [percentage])

  // Calculate the stroke-dasharray for the circular progress
  const radius = 80
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (displayPercentage / 100) * circumference

  // Get color based on percentage
  const getColor = () => {
    if (percentage <= 30) return '#22c55e' // green
    if (percentage <= 60) return '#f97316' // orange
    return '#ef4444' // red
  }

  return (
    <div className="relative w-48 h-48">
      {/* Background circle */}
      <svg className="w-full h-full transform -rotate-90">
        <circle
          cx="96"
          cy="96"
          r={radius}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="12"
        />
        {/* Progress circle */}
        <motion.circle
          cx="96"
          cy="96"
          r={radius}
          fill="none"
          stroke={getColor()}
          strokeWidth="12"
          strokeLinecap="round"
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1.5, ease: 'easeOut' }}
          style={{
            strokeDasharray: circumference,
          }}
        />
      </svg>

      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
          className="text-4xl font-bold text-gray-900"
        >
          {displayPercentage}%
        </motion.span>
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
          className="text-sm text-gray-500 mt-1"
        >
          Risk Score
        </motion.span>
      </div>

      {/* Decorative glow effect */}
      <div
        className="absolute inset-0 rounded-full blur-xl opacity-20"
        style={{ backgroundColor: getColor() }}
      />
    </div>
  )
}
