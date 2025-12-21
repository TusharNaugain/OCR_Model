'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { Loader2, CheckCircle, XCircle, Download, ArrowLeft, FileText } from 'lucide-react';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function ResultsPage() {
    const searchParams = useSearchParams();
    const jobId = searchParams.get('job_id');

    const [status, setStatus] = useState<any>(null);
    const [results, setResults] = useState<any>(null);
    const [error, setError] = useState('');

    useEffect(() => {
        if (!jobId) return;

        const pollStatus = async () => {
            try {
                const response = await axios.get(`${API_URL}/api/ocr/status/${jobId}`);
                setStatus(response.data);

                if (response.data.status === 'completed') {
                    // Fetch results
                    const resultsResponse = await axios.get(`${API_URL}/api/ocr/results/${jobId}`);
                    setResults(resultsResponse.data);
                } else if (response.data.status === 'failed') {
                    setError(response.data.message || 'Processing failed');
                }
            } catch (err: any) {
                setError(err.response?.data?.detail || 'Failed to fetch status');
            }
        };

        // Poll every 2 seconds if still processing
        const interval = setInterval(() => {
            if (status?.status === 'processing' || status?.status === 'pending' || !status) {
                pollStatus();
            }
        }, 2000);

        pollStatus();

        return () => clearInterval(interval);
    }, [jobId, status?.status]);

    const downloadResults = () => {
        if (!results) return;

        const dataStr = JSON.stringify(results.results, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `ocr-results-${jobId}.json`;
        link.click();
        URL.revokeObjectURL(url);
    };

    if (!jobId) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                    <XCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                    <p className="text-xl text-gray-400 mb-6">No job ID provided</p>
                    <Link href="/upload">
                        <button className="px-6 py-3 bg-purple-600 rounded-lg hover:bg-purple-700 transition-colors">
                            Upload Files
                        </button>
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <main className="min-h-screen relative overflow-hidden">
            <div className="absolute top-0 right-1/4 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl" />

            <div className="relative container mx-auto px-6 py-12">
                <Link href="/upload" className="inline-flex items-center text-gray-400 hover:text-white transition-colors mb-8">
                    <ArrowLeft className="w-5 h-5 mr-2" />
                    Upload More Files
                </Link>

                <div className="max-w-3xl mx-auto">
                    {/* Processing Status */}
                    {status && status.status !== 'completed' && !error && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="glass rounded-2xl p-12 text-center"
                        >
                            <Loader2 className="w-16 h-16 mx-auto mb-6 text-purple-400 animate-spin" />
                            <h2 className="text-3xl font-bold mb-4">Processing Your Documents</h2>
                            <p className="text-gray-400 mb-6">{status.message}</p>

                            {/* Progress Bar */}
                            <div className="w-full bg-gray-800 rounded-full h-3 mb-4">
                                <div
                                    className="bg-gradient-to-r from-purple-600 to-blue-600 h-3 rounded-full transition-all duration-500"
                                    style={{ width: `${status.progress}%` }}
                                />
                            </div>

                            <div className="flex justify-between text-sm text-gray-500">
                                <span  >{status.processed_count} / {status.file_count} files</span>
                                <span>{status.progress}%</span>
                            </div>
                        </motion.div>
                    )}

                    {/* Error State */}
                    {error && (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="glass rounded-2xl p-12 text-center border border-red-500/50"
                        >
                            <XCircle className="w-16 h-16 mx-auto mb-6 text-red-500" />
                            <h2 className="text-3xl font-bold mb-4">Processing Failed</h2>
                            <p className="text-gray-400 mb-8">{error}</p>
                            <Link href="/upload">
                                <button className="px-8 py-4 bg-purple-600 rounded-lg hover:bg-purple-700 transition-colors">
                                    Try Again
                                </button>
                            </Link>
                        </motion.div>
                    )}

                    {/* Success State */}
                    {results && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                        >
                            <div className="glass rounded-2xl p-12 text-center mb-8">
                                <CheckCircle className="w-16 h-16 mx-auto mb-6 text-green-500" />
                                <h2 className="text-3xl font-bold mb-4">Processing Complete!</h2>
                                <p className="text-gray-400 mb-8">
                                    Successfully processed {results.results.length} file(s)
                                </p>
                                <button
                                    onClick={downloadResults}
                                    className="px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg font-semibold hover:shadow-2xl hover:shadow-purple-500/50 transition-all duration-300 inline-flex items-center gap-3"
                                >
                                    <Download className="w-5 h-5" />
                                    Download Results (JSON)
                                </button>
                            </div>

                            {/* Results Preview */}
                            <div className="space-y-4">
                                <h3 className="text-2xl font-bold mb-4">Extracted Text:</h3>
                                {results.results.map((result: any, idx: number) => (
                                    <div key={idx} className="glass rounded-xl p-6">
                                        <div className="flex items-center gap-3 mb-4">
                                            <FileText className="w-5 h-5 text-purple-400" />
                                            <h4 className="font-semibold">{result.file || `Document ${idx + 1}`}</h4>
                                        </div>
                                        {result.extracted_text?.lines && (
                                            <div className="bg-black/50 rounded-lg p-4 max-h-60 overflow-y-auto">
                                                <pre className="text-sm text-gray-300 whitespace-pre-wrap font-mono">
                                                    {result.extracted_text.lines.join('\n')}
                                                </pre>
                                            </div>
                                        )}
                                        {result.extracted_text?.confidence && (
                                            <div className="mt-3 text-sm text-gray-500">
                                                Confidence: {(result.extracted_text.confidence * 100).toFixed(1)}%
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    )}
                </div>
            </div>
        </main>
    );
}
