import { motion } from 'framer-motion'

export default function ProgressBar({ current, total }) {
  const percentage = (current / total) * 100

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm font-medium text-gray-600">
          Step {current} of {total}
        </span>
        <span className="text-sm text-gray-500">
          {Math.round(percentage)}% complete
        </span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className="h-full bg-gradient-to-r from-teal-400 to-teal-600 rounded-full"
        />
      </div>
    </div>
  )
}
