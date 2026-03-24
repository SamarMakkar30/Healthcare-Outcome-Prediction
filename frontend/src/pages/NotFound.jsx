import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Home, ArrowLeft } from 'lucide-react'
import Layout from '../components/Layout'

export default function NotFound() {
  const navigate = useNavigate()

  return (
    <Layout>
      <div className="min-h-screen flex items-center justify-center px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <div className="text-8xl mb-6">🔍</div>
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Page Not Found</h1>
          <p className="text-gray-600 mb-8 max-w-md">
            The page you're looking for doesn't exist or has been moved.
          </p>
          <div className="flex gap-4 justify-center">
            <button
              onClick={() => navigate(-1)}
              className="btn-secondary flex items-center gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Go Back
            </button>
            <button
              onClick={() => navigate('/')}
              className="btn-primary flex items-center gap-2"
            >
              <Home className="w-4 h-4" />
              Home
            </button>
          </div>
        </motion.div>
      </div>
    </Layout>
  )
}
