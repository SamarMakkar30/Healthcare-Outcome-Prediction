import { Heart } from 'lucide-react'

export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-teal-50/50 via-white to-blue-50/50">
      {/* Subtle background decoration */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-teal-200/20 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-200/20 rounded-full blur-3xl" />
      </div>

      {/* Main content */}
      <div className="relative z-10">
        {children}
      </div>

      {/* Footer */}
      <footer className="relative z-10 py-6 text-center text-sm text-gray-500">
        <div className="flex items-center justify-center gap-2">
          <Heart className="w-4 h-4 text-teal-500" />
          <span>AI Health Risk Assessment</span>
        </div>
        <p className="mt-1">For educational purposes only</p>
      </footer>
    </div>
  )
}
