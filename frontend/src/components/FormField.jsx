import { HelpCircle } from 'lucide-react'

export default function FormField({ field, value, onChange, error }) {
  const { name, label, type, options, min, max, step, unit, hint } = field

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="label flex items-center gap-2">
          {label}
          {hint && (
            <span className="group relative">
              <HelpCircle className="w-4 h-4 text-gray-400 cursor-help" />
              <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                {hint}
              </span>
            </span>
          )}
        </label>
        {type === 'slider' && (
          <span className="text-sm font-semibold text-teal-600">
            {value} {unit}
          </span>
        )}
      </div>

      {type === 'slider' && (
        <SliderInput
          value={value}
          onChange={onChange}
          min={min}
          max={max}
          step={step || 1}
        />
      )}

      {type === 'toggle' && (
        <ToggleInput
          value={value}
          onChange={onChange}
          options={options}
        />
      )}

      {type === 'select' && (
        <SelectInput
          value={value}
          onChange={onChange}
          options={options}
        />
      )}

      {type === 'number' && (
        <input
          type="number"
          value={value || ''}
          onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
          min={min}
          max={max}
          step={step || 1}
          className="input-field"
        />
      )}

      {error && (
        <p className="text-sm text-red-500">{error}</p>
      )}
    </div>
  )
}

function SliderInput({ value, onChange, min, max, step }) {
  const percentage = ((value - min) / (max - min)) * 100

  return (
    <div className="relative pt-1">
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider-track"
        style={{
          background: `linear-gradient(to right, #14b8a6 0%, #14b8a6 ${percentage}%, #e5e7eb ${percentage}%, #e5e7eb 100%)`
        }}
      />
      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

function ToggleInput({ value, onChange, options }) {
  return (
    <div className="flex gap-3">
      {options.map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => onChange(option)}
          className={`flex-1 py-3 px-4 rounded-xl font-medium transition-all duration-200 ${
            value === option
              ? 'bg-teal-500 text-white shadow-lg shadow-teal-500/30'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          {option}
        </button>
      ))}
    </div>
  )
}

function SelectInput({ value, onChange, options }) {
  return (
    <select
      value={value || ''}
      onChange={(e) => onChange(e.target.value)}
      className="input-field appearance-none bg-white cursor-pointer"
    >
      <option value="" disabled>Select an option</option>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )
}
