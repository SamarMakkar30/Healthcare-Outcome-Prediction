import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, ChevronRight } from 'lucide-react'
import Layout from '../components/Layout'
import { useAssessment } from '../context/AssessmentContext'
import { CONDITIONS } from '../config/conditions'

export default function SelectCondition() {
  const navigate = useNavigate()
  const { setSelectedCondition } = useAssessment()

  const handleSelect = (conditionId) => {
    setSelectedCondition(conditionId)
    navigate(`/assess/${conditionId}`)
  }

  const conditionList = Object.values(CONDITIONS)

  return (
    <Layout>
      <div className="min-h-screen py-8 px-4">
        <div className="max-w-4xl mx-auto">
          {/* Back Button */}
          <motion.button
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            onClick={() => navigate('/')}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-8 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back to Home</span>
          </motion.button>

          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              What would you like to assess?
            </h1>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Choose a health condition to evaluate your risk factors. 
              Each assessment takes about 2-3 minutes.
            </p>
          </motion.div>

          {/* Condition Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {conditionList.map((condition, index) => (
              <motion.div
                key={condition.id}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <button
                  onClick={() => handleSelect(condition.id)}
                  className="w-full text-left card-hover group"
                >
                  {/* Icon */}
                  <div className={`w-16 h-16 rounded-2xl ${condition.lightBg} flex items-center justify-center text-3xl mb-4 group-hover:scale-110 transition-transform`}>
                    {condition.icon}
                  </div>

                  {/* Category Badge */}
                  <span className="inline-block px-3 py-1 rounded-full bg-gray-100 text-xs font-medium text-gray-600 mb-3">
                    {condition.category}
                  </span>

                  {/* Title */}
                  <h2 className="text-xl font-bold text-gray-900 mb-1">
                    {condition.name}
                  </h2>
                  <p className="text-sm text-gray-500 mb-4">
                    {condition.subtitle}
                  </p>

                  {/* Description */}
                  <p className="text-gray-600 text-sm mb-4 line-clamp-2">
                    {condition.description}
                  </p>

                  {/* Action */}
                  <div className="flex items-center text-teal-600 font-medium group-hover:text-teal-700">
                    <span>Start Assessment</span>
                    <ChevronRight className="w-5 h-5 ml-1 group-hover:translate-x-1 transition-transform" />
                  </div>
                </button>
              </motion.div>
            ))}
          </div>

          {/* Info Box */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="mt-12 p-6 rounded-2xl bg-gradient-to-r from-teal-50 to-blue-50 border border-teal-100"
          >
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-full bg-teal-100 flex items-center justify-center flex-shrink-0">
                <span className="text-xl">💡</span>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">How it works</h3>
                <p className="text-sm text-gray-600">
                  Our AI analyzes multiple health factors to estimate your risk level. 
                  You'll answer questions about your lifestyle, health metrics, and medical history. 
                  Results are instant and include personalized recommendations.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </Layout>
  )
}
