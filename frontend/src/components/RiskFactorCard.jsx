import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

export default function RiskFactorCard({ factor, index }) {
  const impact = factor.impact?.toLowerCase() || 'neutral'
  
  const getImpactConfig = () => {
    switch (impact) {
      case 'increases':
      case 'positive':
        return {
          icon: <TrendingUp className="w-4 h-4" />,
          color: 'text-red-500',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200',
          label: 'Increases risk'
        }
      case 'decreases':
      case 'negative':
        return {
          icon: <TrendingDown className="w-4 h-4" />,
          color: 'text-green-500',
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200',
          label: 'Decreases risk'
        }
      default:
        return {
          icon: <Minus className="w-4 h-4" />,
          color: 'text-gray-500',
          bgColor: 'bg-gray-50',
          borderColor: 'border-gray-200',
          label: 'Neutral'
        }
    }
  }

  const config = getImpactConfig()

  // Format feature name for display
  const formatFeatureName = (name) => {
    return name
      .replace(/_/g, ' ')
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase())
      .trim()
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className={`p-4 rounded-xl ${config.bgColor} border ${config.borderColor}`}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h4 className="font-medium text-gray-900">
            {formatFeatureName(factor.feature || factor.name)}
          </h4>
          {factor.value !== undefined && (
            <p className="text-sm text-gray-500 mt-1">
              Value: {typeof factor.value === 'number' ? factor.value.toFixed(2) : String(factor.value)}
            </p>
          )}
        </div>
        <div className={`flex items-center gap-1 ${config.color}`}>
          {config.icon}
          <span className="text-xs font-medium">{config.label}</span>
        </div>
      </div>
      
      {/* Importance bar */}
      {factor.importance !== undefined && (
        <div className="mt-3">
          <div className="h-1.5 bg-white rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(factor.importance * 100, 100)}%` }}
              transition={{ duration: 0.5, delay: 0.3 + index * 0.1 }}
              className={`h-full rounded-full ${
                impact === 'increases' ? 'bg-red-400' : 
                impact === 'decreases' ? 'bg-green-400' : 'bg-gray-400'
              }`}
            />
          </div>
        </div>
      )}

      {factor.clinical_note && (
        <p className="text-xs text-gray-600 mt-2">{factor.clinical_note}</p>
      )}
    </motion.div>
  )
}
