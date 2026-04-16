import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Image as ImageIcon, X } from 'lucide-react'

export default function UploadZone({ onFile, loading }) {
  const [preview, setPreview] = useState(null)

  const onDrop = useCallback(accepted => {
    if (!accepted.length) return
    const file = accepted[0]
    setPreview(URL.createObjectURL(file))
    onFile(file)
  }, [onFile])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [], 'image/webp': [] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
    disabled: loading,
  })

  const clear = e => {
    e.stopPropagation()
    setPreview(null)
    onFile(null)
  }

  return (
    <div
      {...getRootProps()}
      className={`relative border-2 border-dashed rounded-xl transition-all cursor-pointer
        ${isDragActive ? 'border-cyan-300/60 bg-cyan-400/10' : 'border-white/20 hover:border-white/40 bg-slate-900/20'}
        ${loading ? 'opacity-60 pointer-events-none' : ''}
      `}
    >
      <input {...getInputProps()} />

      {preview ? (
        <div className="relative">
          <img
            src={preview}
            alt="Preview"
            className="w-full max-h-72 object-contain rounded-xl p-2"
          />
          {!loading && (
            <button
              onClick={clear}
              className="absolute top-3 right-3 bg-slate-900/70 text-slate-200 rounded-full p-1 shadow-md
                         hover:bg-red-500/20 hover:text-red-200 transition-colors"
            >
              <X size={16} />
            </button>
          )}
          {loading && (
            <div className="absolute inset-0 bg-slate-950/60 flex items-center justify-center rounded-xl">
              <div className="flex flex-col items-center gap-2">
                <div className="w-8 h-8 border-2 border-cyan-300 border-t-transparent
                                rounded-full animate-spin" />
                <span className="text-sm text-slate-200 font-medium">Analysing...</span>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-16 px-8 gap-3">
          <div className="p-4 bg-white/5 rounded-full border border-white/10">
            {isDragActive
              ? <ImageIcon size={28} className="text-cyan-300" />
              : <Upload size={28} className="text-slate-400" />
            }
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-slate-200">
              {isDragActive ? 'Drop image here' : 'Drag and drop an image'}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              or click to browse · JPEG, PNG, WebP · max 10MB
            </p>
          </div>
        </div>
      )}
    </div>
  )
}