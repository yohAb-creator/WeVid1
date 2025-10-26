import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import { promises as fs } from 'fs'
import path from 'path'
import os from 'os'

export async function POST(request: NextRequest) {
  try {
    const { url, interests } = await request.json()

    if (!url || !interests) {
      return NextResponse.json(
        { error: 'URL and interests are required' },
        { status: 400 }
      )
    }

    console.log('Starting Python-based audio analysis for URL:', url)
    
    // Create temporary files for communication with Python script
    const tempDir = os.tmpdir()
    const inputFile = path.join(tempDir, `input_${Date.now()}.json`)
    const outputFile = path.join(tempDir, `output_${Date.now()}.json`)
    
    // Prepare input data for Python script
    const inputData = {
      url,
      interests,
      output_file: outputFile
    }
    
    try {
      // Write input data to temporary file
      await fs.writeFile(inputFile, JSON.stringify(inputData, null, 2))
      console.log('Input data written to:', inputFile)
      
      // Run Python bridge script
      const pythonScript = path.join(process.cwd(), 'analyze_youtube_bridge.py')
      console.log('Running Python script:', pythonScript)
      
      const pythonProcess = spawn('python', [pythonScript, inputFile], {
        cwd: process.cwd(),
        stdio: ['pipe', 'pipe', 'pipe']
      })
      
      let stdout = ''
      let stderr = ''
      
      // Capture Python output
      pythonProcess.stdout.on('data', (data) => {
        const output = data.toString()
        stdout += output
        console.log('Python stdout:', output.trim())
      })
      
      pythonProcess.stderr.on('data', (data) => {
        const error = data.toString()
        stderr += error
        console.error('Python stderr:', error.trim())
      })
      
      // Wait for Python process to complete
      const exitCode = await new Promise<number>((resolve, reject) => {
        pythonProcess.on('close', (code) => {
          console.log(`Python process exited with code: ${code}`)
          resolve(code)
        })
        
        pythonProcess.on('error', (error) => {
          console.error('Python process error:', error)
          reject(error)
        })
      })
      
      if (exitCode !== 0) {
        throw new Error(`Python script failed with exit code ${exitCode}. Stderr: ${stderr}`)
      }
      
      // Read results from output file
      const outputData = await fs.readFile(outputFile, 'utf8')
      const results = JSON.parse(outputData)
      
      console.log('Python analysis completed successfully')
      console.log(`Found ${results.segments?.length || 0} segments`)
      
      // Clean up temporary files
      try {
        await fs.unlink(inputFile)
        await fs.unlink(outputFile)
        console.log('Temporary files cleaned up')
      } catch (cleanupError) {
        console.warn('Failed to clean up temporary files:', cleanupError)
      }
      
      return NextResponse.json({
        success: true,
        segments: results.segments || [],
        videoInfo: results.videoInfo || {},
        processingTime: results.processingTime || 0,
        source: 'assemblyai_python',
        pythonOutput: stdout,
        originalSegmentCount: results.originalSegmentCount,
        convertedSegmentCount: results.convertedSegmentCount
      })
      
    } catch (error) {
      console.error('Python analysis error:', error)
      
      // Clean up temporary files on error
      try {
        await fs.unlink(inputFile).catch(() => {})
        await fs.unlink(outputFile).catch(() => {})
      } catch (cleanupError) {
        console.warn('Failed to clean up temporary files after error:', cleanupError)
      }
      
      return NextResponse.json(
        { 
          success: false,
          error: `Failed to analyze video with Python backend: ${error instanceof Error ? error.message : String(error)}`,
          pythonStderr: stderr,
          pythonStdout: stdout
        },
        { status: 500 }
      )
    }
    
  } catch (error) {
    console.error('API route error:', error)
    return NextResponse.json(
      { 
        success: false,
        error: 'Internal server error' 
      },
      { status: 500 }
    )
  }
}
