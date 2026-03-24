import { motion } from 'framer-motion'
import { CheckCircle, AlertCircle, Info, ChevronRight } from 'lucide-react'

export default function RecommendationCard({ recommendation, index }) {
  // Handle both object and string recommendations
  const rec = typeof recommendation === 'string' 
    ? { advice: recommendation, category: 'General', priority: 'normal' }
    : recommendation

  const getPriorityConfig = () => {
    switch (rec.priority?.toLowerCase()) {
      case 'urgent':
      case 'high':
        return {
          icon: <AlertCircle className="w-5 h-5" />,
          color: 'text-red-500',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200',
          badge: 'Urgent'
        }
      case 'important':
      case 'medium':
        return {
          icon: <Info className="w-5 h-5" />,
          color: 'text-amber-500',
          bgColor: 'bg-amber-50',
          borderColor: 'border-amber-200',
          badge: 'Important'
        }
      default:
        return {
          icon: <CheckCircle className="w-5 h-5" />,
          color: 'text-teal-500',
          bgColor: 'bg-teal-50',
          borderColor: 'border-teal-200',
          badge: null
        }
    }
  }

  const config = getPriorityConfig()

  // Get category icon
  const getCategoryIcon = () => {
    const category = rec.category?.toLowerCase() || ''
    if (category.includes('diet') || category.includes('nutrition')) return '🥗'
    if (category.includes('exercise') || category.includes('physical')) return '🏃'
    if (category.includes('medical') || category.includes('doctor')) return '🩺'
    if (category.includes('lifestyle')) return '🌟'
    if (category.includes('mental') || category.includes('stress')) return '🧘'
    if (category.includes('sleep')) return '😴'
    if (category.includes('urgent')) return '⚠️'
    return '💡'
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      className={`p-5 rounded-xl ${config.bgColor} border ${config.borderColor} hover:shadow-md transition-shadow`}
    >
      <div className="flex gap-4">
        {/* Icon */}
        <div className="flex-shrink-0">
          <span className="text-2xl">{getCategoryIcon()}</span>
        </div>

        {/* Content */}
        <div className="flex-1">
          <div className="flex items-start justify-between gap-2">
            <div>
              {rec.category && (
                <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                  {rec.category}
                </span>
              )}
              {config.badge && (
                <span className={`ml-2 px-2 py-0.5 text-xs font-semibold rounded-full ${config.color} bg-white`}>
                  {config.badge}
                </span>
              )}
            </div>
          </div>

          <p className="text-gray-800 mt-1 leading-relaxed">
            {rec.advice}
          </p>

          {rec.evidence_level && (
            <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
              <Info className="w-3 h-3" />
              Evidence: {rec.evidence_level}
            </p>
          )}
        </div>

        {/* Arrow indicator */}
        <div className={`flex-shrink-0 ${config.color}`}>
          {config.icon}
        </div>
      </div>
    </motion.div>
  )
}
