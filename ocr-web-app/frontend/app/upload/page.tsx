'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { Upload, FileText, Image, X, ArrowLeft, Loader2 } from 'lucide-react';
import Link from 'next/link';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000');

export default function UploadPage() {
    const router = useRouter();
    const [files, setFiles] = useState<File[]>([]);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState('');

    const onDrop = useCallback((acceptedFiles: File[]) => {
        setFiles(prev => [...prev, ...acceptedFiles]);
        setError('');
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'application/pdf': ['.pdf'],
            'image/*': ['.png', '.jpg', '.jpeg']
        },
        multiple: true
    });

    const removeFile = (index: number) => {
        setFiles(files.filter((_, i) => i !== index));
    };

    const handleUpload = async () => {
        if (files.length === 0) {
            setError('Please select at least one file');
            return;
        }

        setUploading(true);
        setError('');

        try {
            const formData = new FormData();
            files.forEach(file => {
                formData.append('files', file);
            });

            const response = await axios.post(`${API_URL}/api/ocr/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            const { job_id } = response.data;

            // Redirect to processing page
            router.push(`/results?job_id=${job_id}`);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Upload failed. Please try again.');
            setUploading(false);
        }
    };

    return (
        <main className="min-h-screen relative overflow-hidden">
            <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl" />

            <div className="relative container mx-auto px-6 py-12">
                {/* Header */}
                <div className="mb-12">
                    <Link href="/" className="inline-flex items-center text-gray-400 hover:text-white transition-colors mb-6">
                        <ArrowLeft className="w-5 h-5 mr-2" />
                        Back to Home
                    </Link>

                    <h1 className="text-5xl font-bold mb-4">
                        Upload Your <span className="gradient-text">Documents</span>
                    </h1>
                    <p className="text-xl text-gray-400">
                        Support for PDF, PNG, JPG, and JPEG files. Process up to 1,500 pages daily for free.
                    </p>
                </div>

                {/* Upload Zone */}
                <div className="max-w-4xl mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                    >
                        <div
                            {...getRootProps()}
                            className={`glass rounded-3xl p-12 border-2 border-dashed transition-all duration-300 cursor-pointer
                ${isDragActive ? 'border-purple-500 bg-purple-500/10' : 'border-white/20 hover:border-purple-500/50'}`}
                        >
                            <input {...getInputProps()} />
                            <div className="text-center">
                                <Upload className="w-16 h-16 mx-auto mb-6 text-purple-400" />
                                <h3 className="text-2xl font-semibold mb-2">
                                    {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
                                </h3>
                                <p className="text-gray-400 mb-4">or click to browse</p>
                                <div className="flex gap-4 justify-center text-sm text-gray-500">
                                    <span className="flex items-center gap-2">
                                        <FileText className="w-4 h-4" /> PDF
                                    </span>
                                    <span className="flex items-center gap-2">
                                        <Image className="w-4 h-4" /> PNG, JPG
                                    </span>
                                </div>
                            </div>
                        </div>
                    </motion.div>

                    {/* File List */}
                    {files.length > 0 && (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="mt-8"
                        >
                            <h3 className="text-xl font-semibold mb-4">
                                Selected Files ({files.length})
                            </h3>
                            <div className="space-y-2">
                                {files.map((file, idx) => (
                                    <div key={idx} className="glass rounded-lg p-4 flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            {file.type.includes('pdf') ? (
                                                <FileText className="w-5 h-5 text-red-400" />
                                            ) : (
                                                <Image className="w-5 h-5 text-blue-400" />
                                            )}
                                            <span>{file.name}</span>
                                            <span className="text-sm text-gray-500">
                                                ({(file.size / 1024 / 1024).toFixed(2)} MB)
                                            </span>
                                        </div>
                                        <button
                                            onClick={() => removeFile(idx)}
                                            className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
                                        >
                                            <X className="w-5 h-5 text-red-400" />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </motion.div>
                    )}

                    {/* Error Message */}
                    {error && (
                        <div className="mt-6 glass rounded-lg p-4 border border-red-500/50 bg-red-500/10">
                            <p className="text-red-400">{error}</p>
                        </div>
                    )}

                    {/* Upload Button */}
                    {files.length > 0 && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="mt-8 flex justify-center"
                        >
                            <button
                                onClick={handleUpload}
                                disabled={uploading}
                                className="px-10 py-5 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg font-semibold text-xl hover:shadow-2xl hover:shadow-purple-500/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3"
                            >
                                {uploading ? (
                                    <>
                                        <Loader2 className="w-6 h-6 animate-spin" />
                                        Uploading...
                                    </>
                                ) : (
                                    <>
                                        <Upload className="w-6 h-6" />
                                        Start Processing
                                    </>
                                )}
                            </button>
                        </motion.div>
                    )}
                </div>

                {/* Info Cards */}
                <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto mt-16">
                    {[
                        { title: "Fast Processing", desc: "1-2 seconds per page" },
                        { title: "Multi-Format", desc: "PDF, PNG, JPG, JPEG supported" },
                        { title: "AI Enhanced", desc: "Gemini-powered accuracy" }
                    ].map((item, idx) => (
                        <div key={idx} className="glass rounded-xl p-6 text-center">
                            <h4 className="font-semibold mb-2">{item.title}</h4>
                            <p className="text-sm text-gray-400">{item.desc}</p>
                        </div>
                    ))}
                </div>
            </div>
        </main>
    );
}
