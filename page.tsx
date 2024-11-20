'use client'

import { useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Upload, FileText, Download } from 'lucide-react'

export default function PDFToMarkdownConverter() {
  const [file, setFile] = useState<File | null>(null)
  const [convertedText, setConvertedText] = useState<string>('')
  const [isConverting, setIsConverting] = useState(false)
  const [error, setError] = useState<string>('')

  const onDrop = (acceptedFiles: File[]) => {
    if (acceptedFiles[0].type === 'application/pdf') {
      setFile(acceptedFiles[0])
      setError('')
    } else {
      setError('Please upload a PDF file')
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
  })

  const handleConvert = async () => {
    if (!file) return

    setIsConverting(true)
    setError('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/convert', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Conversion failed')
      }

      const result = await response.json()
      setConvertedText(result.markdown)
    } catch (err) {
      setError('Failed to convert PDF. Please try again.')
    } finally {
      setIsConverting(false)
    }
  }

  const handleDownload = () => {
    const blob = new Blob([convertedText], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${file?.name.replace('.pdf', '')}.md` || 'converted.md'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">PDF to Markdown Converter</h1>
      <Card className="p-6">
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer ${
            isDragActive ? 'border-primary' : 'border-gray-300'
          }`}
        >
          <input {...getInputProps()} />
          {file ? (
            <p>{file.name}</p>
          ) : (
            <p>Drag and drop a PDF file here, or click to select a file</p>
          )}
          <Upload className="mx-auto mt-4" size={48} />
        </div>
        {error && <p className="text-red-500 mt-2">{error}</p>}
        <div className="mt-4">
          <Button onClick={handleConvert} disabled={!file || isConverting}>
            {isConverting ? 'Converting...' : 'Convert to Markdown'}
          </Button>
        </div>
        {convertedText && (
          <div className="mt-6">
            <Label htmlFor="markdown">Converted Markdown:</Label>
            <div className="relative mt-2">
              <Input
                id="markdown"
                value={convertedText}
                readOnly
                className="h-48 font-mono"
              />
              <Button
                onClick={handleDownload}
                className="absolute top-2 right-2"
                size="icon"
              >
                <Download size={16} />
              </Button>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}