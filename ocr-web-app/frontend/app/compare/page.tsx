'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FileUp, ArrowLeft, CheckCircle, AlertCircle, FileText, Database, Scale } from 'lucide-react';
import Link from 'next/link';
import axios from 'axios';

export default function ComparePage() {
    const [refFile, setRefFile] = useState<File | null>(null);
    const [docFile, setDocFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: 'ref' | 'doc') => {
        if (e.target.files && e.target.files[0]) {
            if (type === 'ref') setRefFile(e.target.files[0]);
            else setDocFile(e.target.files[0]);
            setError(null);
        }
    };

    const handleCompare = async () => {
        if (!refFile || !docFile) {
            setError("Please upload both files.");
            return;
        }

        setLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append('reference_file', refFile);
        formData.append('document_file', docFile);

        try {
            // Assuming API is proxied or direct. Adjust URL if needed.
            // Since page.tsx didn't have API URL, I'll assume relative path /api or localhost:8000
            // Usually development setup has proxy in next.config.mjs or direct.
            // Let's try direct to backend port 8000 if typical setup, or /api if proxied.
            // Given 'api:app' runs on 8000, and Next on 3000.
            const apiUrl = 'http://localhost:8000/api/ocr/compare';

            const response = await axios.post(apiUrl, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            setResult(response.data.comparison);
        } catch (err: any) {
            console.error(err);
            setError(err.response?.data?.error || "Comparison failed. Ensure backend is running.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="min-h-screen bg-black text-white p-6 relative overflow-x-hidden">
            {/* Background blobs */}
            <div className="absolute top-0 left-0 w-96 h-96 bg-purple-900/20 rounded-full blur-3xl pointer-events-none" />
            <div className="absolute bottom-0 right-0 w-96 h-96 bg-blue-900/20 rounded-full blur-3xl pointer-events-none" />

            <div className="max-w-6xl mx-auto relative z-10">
                {/* Header */}
                <div className="flex items-center mb-8">
                    <Link href="/" className="mr-4 p-2 rounded-full hover:bg-white/10 transition-colors">
                        <ArrowLeft className="w-6 h-6 text-gray-400" />
                    </Link>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                        Document Comparison
                    </h1>
                </div>

                {/* Upload Area */}
                <div className="grid md:grid-cols-2 gap-8 mb-8">
                    {/* Reference File */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="glass rounded-2xl p-8 border border-white/10"
                    >
                        <div className="flex items-center mb-4 text-purple-400">
                            <Database className="w-6 h-6 mr-2" />
                            <h2 className="text-xl font-semibold">1. Reference Data</h2>
                        </div>
                        <p className="text-gray-400 text-sm mb-6">Upload Excel (.xlsx) or CSV file with the correct data.</p>

                        <div className="relative group">
                            <input
                                type="file"
                                accept=".csv, .xlsx, .xls"
                                onChange={(e) => handleFileChange(e, 'ref')}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                            />
                            <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${refFile ? 'border-green-500/50 bg-green-500/10' : 'border-gray-600 hover:border-purple-500 hover:bg-white/5'}`}>
                                {refFile ? (
                                    <div className="text-green-400 font-medium flex flex-col items-center">
                                        <CheckCircle className="w-8 h-8 mb-2" />
                                        {refFile.name}
                                    </div>
                                ) : (
                                    <div className="text-gray-400 flex flex-col items-center">
                                        <FileUp className="w-8 h-8 mb-2" />
                                        <span>Click to upload Excel/CSV</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </motion.div>

                    {/* Document File */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="glass rounded-2xl p-8 border border-white/10"
                    >
                        <div className="flex items-center mb-4 text-blue-400">
                            <FileText className="w-6 h-6 mr-2" />
                            <h2 className="text-xl font-semibold">2. Scanned Document</h2>
                        </div>
                        <p className="text-gray-400 text-sm mb-6">Upload PDF or Image to OCR and compare.</p>

                        <div className="relative group">
                            <input
                                type="file"
                                accept=".pdf, .png, .jpg, .jpeg"
                                onChange={(e) => handleFileChange(e, 'doc')}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                            />
                            <div className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${docFile ? 'border-blue-500/50 bg-blue-500/10' : 'border-gray-600 hover:border-blue-500 hover:bg-white/5'}`}>
                                {docFile ? (
                                    <div className="text-blue-400 font-medium flex flex-col items-center">
                                        <CheckCircle className="w-8 h-8 mb-2" />
                                        {docFile.name}
                                    </div>
                                ) : (
                                    <div className="text-gray-400 flex flex-col items-center">
                                        <FileUp className="w-8 h-8 mb-2" />
                                        <span>Click to upload PDF/Image</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </motion.div>
                </div>

                {/* Action */}
                <div className="text-center mb-12">
                    <button
                        onClick={handleCompare}
                        disabled={loading || !refFile || !docFile}
                        className="group relative px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl font-bold text-lg hover:shadow-2xl hover:shadow-purple-500/50 disabled:opacity-50 disabled:cursor-not-allowed transition-all active:scale-95"
                    >
                        {loading ? (
                            <span className="flex items-center">
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Processing OCR & Matching...
                            </span>
                        ) : (
                            <span className="flex items-center">
                                <Scale className="mr-2 w-5 h-5 group-hover:rotate-12 transition-transform" />
                                Compare Documents
                            </span>
                        )}
                    </button>
                    {error && (
                        <div className="mt-4 text-red-400 flex items-center justify-center bg-red-900/20 p-3 rounded-lg max-w-md mx-auto">
                            <AlertCircle className="w-5 h-5 mr-2" />
                            {error}
                        </div>
                    )}
                </div>

                {/* Results */}
                {result && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="glass rounded-3xl p-8 border border-white/10"
                    >
                        <div className="flex flex-col md:flex-row justify-between items-center mb-8 border-b border-white/10 pb-6">
                            <div>
                                <h2 className="text-2xl font-bold mb-1">Comparison Result</h2>
                                <p className="text-gray-400 text-sm">
                                    {result.match_found
                                        ? `Matched with Row #${result.matched_row_index + 1} in reference file.`
                                        : "No strong match found in reference file."}
                                </p>
                            </div>
                            <div className="mt-4 md:mt-0 flex items-center bg-black/40 px-6 py-3 rounded-xl border border-white/5">
                                <span className="text-gray-400 mr-3">Overall Match:</span>
                                <span className={`text-3xl font-bold ${result.overall_match_score > 90 ? 'text-green-400' : result.overall_match_score > 70 ? 'text-yellow-400' : 'text-red-400'}`}>
                                    {result.overall_match_score}%
                                </span>
                            </div>
                        </div>

                        <div className="overflow-x-auto">
                            <table className="w-full text-left border-collapse">
                                <thead>
                                    <tr className="border-b border-gray-700 text-gray-400 text-sm">
                                        <th className="py-3 px-4">Field Name</th>
                                        <th className="py-3 px-4">OCR Value (Document)</th>
                                        <th className="py-3 px-4">Reference Value (File)</th>
                                        <th className="py-3 px-4 text-right">Match</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {result.details && result.details.map((item: any, idx: number) => (
                                        <tr key={idx} className={`border-b border-white/5 hover:bg-white/5 transition-colors ${item.is_match ? '' : 'bg-red-500/5'}`}>
                                            <td className="py-3 px-4 font-mono text-sm text-gray-300">
                                                {item.field}
                                                <div className="text-xs text-gray-500">{item.reference_column}</div>
                                            </td>
                                            <td className="py-3 px-4 text-white">{item.ocr_value || <span className="text-gray-600 italic">Empty</span>}</td>
                                            <td className="py-3 px-4 text-gray-300">{item.reference_value || <span className="text-gray-600 italic">Empty</span>}</td>
                                            <td className="py-3 px-4 text-right">
                                                <div className="inline-flex items-center">
                                                    <span className={`font-bold mr-2 ${item.match_score > 90 ? 'text-green-400' : item.match_score > 60 ? 'text-yellow-400' : 'text-red-400'}`}>
                                                        {item.match_score}%
                                                    </span>
                                                    {item.is_match ? <CheckCircle className="w-4 h-4 text-green-500" /> : <AlertCircle className="w-4 h-4 text-red-500" />}
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </motion.div>
                )}
            </div>
        </main>
    );
}
