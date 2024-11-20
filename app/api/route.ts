import { NextRequest, NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import path from 'path'
import os from 'os'
import pdf from 'pdf-parse'

export async function POST(request: NextRequest) {
  const formData = await request.formData()
  const file = formData.get('file') as File

  if (!file) {
    return NextResponse.json({ error: 'No file uploaded' }, { status: 400 })
  }

  const buffer = Buffer.from(await file.arrayBuffer())
  const tempDir = os.tmpdir()
  const tempFilePath = path.join(tempDir, file.name)

  try {
    await fs.writeFile(tempFilePath, buffer)
    const pdfBuffer = await fs.readFile(tempFilePath)
    const data = await pdf(pdfBuffer)

    // Simple conversion: just use the raw text as Markdown
    const markdown = data.text

    await fs.unlink(tempFilePath)

    return NextResponse.json({ markdown })
  } catch (error) {
    console.error('Error converting PDF:', error)
    return NextResponse.json({ error: 'Failed to convert PDF' }, { status: 500 })
  }
}