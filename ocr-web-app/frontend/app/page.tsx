'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Scan, Zap, Shield, Sparkles, FileText, Image, Upload, ArrowRight, Building2, Scale, CreditCard, Heart, FileSearch, Package, Users, CheckCircle2 } from 'lucide-react';

export default function Home() {
  const features = [
    {
      icon: <Scan className="w-8 h-8" />,
      title: "AI-Powered OCR",
      description: "Extract text from PDFs and images with 95%+ accuracy using advanced AI"
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: "Lightning Fast",
      description: "Process documents in seconds with parallel batch processing"
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Secure & Private",
      description: "Your documents are processed locally and deleted after completion"
    },
    {
      icon: <Sparkles className="w-8 h-8" />,
      title: "Gemini Enhanced",
      description: "AI-powered error correction and context-aware text understanding"
    }
  ];

  const documentTypes = [
    {
      icon: <CreditCard className="w-6 h-6" />,
      title: "Financial Documents",
      description: "Invoices, receipts, checks, bank statements",
      features: ["Extract amounts", "Line items", "Vendor details", "Auto-validation"]
    },
    {
      icon: <FileText className="w-6 h-6" />,
      title: "ID Cards & Passports",
      description: "Driver's licenses, passports, national IDs",
      features: ["MRZ parsing", "Personal info", "Dates extraction", "Face detection"]
    },
    {
      icon: <FileSearch className="w-6 h-6" />,
      title: "Forms & Applications",
      description: "Tax forms, surveys, employee records",
      features: ["Field detection", "Checkboxes", "Signatures", "Dynamic fields"]
    },
    {
      icon: <Scale className="w-6 h-6" />,
      title: "Legal Documents",
      description: "Contracts, deeds, court records",
      features: ["Case numbers", "Parties", "Dates", "Signatures"]
    },
    {
      icon: <Heart className="w-6 h-6" />,
      title: "Healthcare Records",
      description: "Patient records, prescriptions, claims",
      features: ["Patient data", "Medications", "Diagnosis", "Physician info"]
    },
    {
      icon: <Package className="w-6 h-6" />,
      title: "Logistics Documents",
      description: "Shipping labels, purchase orders",
      features: ["Tracking #", "PO numbers", "Addresses", "Carriers"]
    }
  ];

  const industries = [
    {
      icon: <Building2 className="w-8 h-8" />,
      title: "Finance & Accounting",
      description: "Automate invoice processing, receipt management, and expense reporting"
    },
    {
      icon: <Scale className="w-8 h-8" />,
      title: "Legal Firms",
      description: "Digitize contracts, court records, and legal filings for easy searchability"
    },
    {
      icon: <Heart className="w-8 h-8" />,
      title: "Healthcare",
      description: "Process patient records, insurance claims, and prescriptions efficiently"
    },
    {
      icon: <Package className="w-8 h-8" />,
      title: "Logistics & Supply Chain",
      description: "Track shipments, manage inventory, and process delivery documents"
    }
  ];

  const howItWorks = [
    {
      step: "1",
      title: "Upload Document",
      description: "Drag & drop or select PDFs, images (JPG, PNG)"
    },
    {
      step: "2",
      title: "Auto-Classification",
      description: "AI detects document type (invoice, passport, etc.)"
    },
    {
      step: "3",
      title: "Smart Extraction",
      description: "Specialized processor extracts relevant fields"
    },
    {
      step: "4",
      title: "Get Results",
      description: "Receive structured JSON with validated data"
    }
  ];

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Background gradient orbs */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl animate-float" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl animate-float" style={{ animationDelay: '2s' }} />

      {/* Hero Section */}
      <section className="relative container mx-auto px-6 pt-20 pb-32">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center"
        >
          <div className="inline-block mb-6">
            <span className="glass px-4 py-2 rounded-full text-sm font-medium text-purple-300">
              ✨ Powered by Gemini AI
            </span>
          </div>

          <h1 className="text-6xl md:text-8xl font-bold mb-6 leading-tight">
            Transform
            <span className="gradient-text block">Documents into Data</span>
          </h1>

          <p className="text-xl text-gray-400 mb-12 max-w-2xl mx-auto">
            Enterprise-grade OCR with AI-powered classification for 7+ document types.
            Fast, accurate, and free for up to 1,500 pages per day.
          </p>

          <div className="flex gap-4 justify-center">
            <Link href="/upload">
              <button className="group relative px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg font-semibold text-lg hover:shadow-2xl hover:shadow-purple-500/50 transition-all duration-300 animate-glow">
                Start Scanning
                <ArrowRight className="inline-block ml-2 group-hover:translate-x-1 transition-transform" />
              </button>
            </Link>

            <Link href="#how-it-works">
              <button className="px-8 py-4 glass rounded-lg font-semibold text-lg hover:bg-white/10 transition-all duration-300">
                Learn More
              </button>
            </Link>
          </div>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="grid grid-cols-3 gap-8 mt-20 max-w-3xl mx-auto"
        >
          {[
            { value: "7+", label: "Document Types" },
            { value: "2-4x", label: "Faster Processing" },
            { value: "95%+", label: "Accuracy" }
          ].map((stat, idx) => (
            <div key={idx} className="text-center glass rounded-xl p-6">
              <div className="text-4xl font-bold gradient-text mb-2">{stat.value}</div>
              <div className="text-gray-400">{stat.label}</div>
            </div>
          ))}
        </motion.div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="relative container mx-auto px-6 py-20">
        <h2 className="text-4xl md:text-5xl font-bold text-center mb-16">
          How It <span className="gradient-text">Works</span>
        </h2>

        <div className="grid md:grid-cols-4 gap-6 max-w-5xl mx-auto">
          {howItWorks.map((step, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: idx * 0.1 }}
              viewport={{ once: true }}
              className="glass rounded-2xl p-6 text-center"
            >
              <div className="w-12 h-12 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 flex items-center justify-center text-2xl font-bold mx-auto mb-4">
                {step.step}
              </div>
              <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
              <p className="text-gray-400 text-sm">{step.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Document Types Section */}
      <section className="relative container mx-auto px-6 py-20">
        <h2 className="text-4xl md:text-5xl font-bold text-center mb-4">
          Supported <span className="gradient-text">Document Types</span>
        </h2>
        <p className="text-xl text-gray-400 text-center mb-16 max-w-3xl mx-auto">
          Our AI automatically detects document type and extracts specialized fields with validation
        </p>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {documentTypes.map((doc, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: idx * 0.1 }}
              viewport={{ once: true }}
              className="glass rounded-2xl p-6 hover:bg-white/10 transition-all duration-300 group"
            >
              <div className="text-blue-400 mb-4 group-hover:scale-110 transition-transform">
                {doc.icon}
              </div>
              <h3 className="text-xl font-semibold mb-2">{doc.title}</h3>
              <p className="text-gray-400 text-sm mb-4">{doc.description}</p>
              <div className="space-y-2">
                {doc.features.map((feature, fidx) => (
                  <div key={fidx} className="flex items-center text-sm text-gray-300">
                    <CheckCircle2 className="w-4 h-4 text-green-400 mr-2" />
                    {feature}
                  </div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative container mx-auto px-6 py-20">
        <h2 className="text-4xl md:text-5xl font-bold text-center mb-16">
          Why Choose <span className="gradient-text">SmartScan</span>?
        </h2>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: idx * 0.1 }}
              viewport={{ once: true }}
              className="glass rounded-2xl p-6 hover:bg-white/10 transition-all duration-300 group"
            >
              <div className="text-purple-400 mb-4 group-hover:scale-110 transition-transform">
                {feature.icon}
              </div>
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-gray-400 text-sm">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Industries Section */}
      <section className="relative container mx-auto px-6 py-20">
        <h2 className="text-4xl md:text-5xl font-bold text-center mb-4">
          Perfect For <span className="gradient-text">Every Industry</span>
        </h2>
        <p className="text-xl text-gray-400 text-center mb-16 max-w-3xl mx-auto">
          Trusted by businesses across finance, legal, healthcare, and logistics
        </p>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-6xl mx-auto">
          {industries.map((industry, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: idx * 0.1 }}
              viewport={{ once: true }}
              className="glass rounded-2xl p-8 hover:bg-white/10 transition-all duration-300 text-center"
            >
              <div className="text-purple-400 mb-4 flex justify-center">{industry.icon}</div>
              <h3 className="text-xl font-semibold mb-3">{industry.title}</h3>
              <p className="text-gray-400 text-sm">{industry.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Use Cases Section */}
      <section className="relative container mx-auto px-6 py-20 bg-gradient-to-b from-transparent via-purple-900/10 to-transparent">
        <h2 className="text-4xl md:text-5xl font-bold text-center mb-16">
          Real-World <span className="gradient-text">Use Cases</span>
        </h2>

        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          {[
            {
              title: "Automated Invoice Processing",
              description: "Process hundreds of invoices per day with auto-extraction of amounts, dates, vendor info, and line items. Validate totals automatically.",
              example: "Extract: Invoice #, Total ($1,234.56), Tax, Line Items"
            },
            {
              title: "Identity Verification",
              description: "Instantly verify passports and driver's licenses with MRZ parsing. Extract name, DOB, ID numbers, and expiration dates.",
              example: "Extract: Name, ID #, DOB, Nationality, Expiry Date"
            },
            {
              title: "Document Digitization",
              description: "Convert paper archives, historical records, and handwritten notes into searchable digital text for preservation.",
              example: "Books, newspapers, letters, court records"
            },
            {
              title: "Healthcare Data Entry",
              description: "Automate patient record processing, prescription reading, and insurance claim data extraction with medical terminology support.",
              example: "Extract: Patient name, Medications, Diagnosis codes"
            }
          ].map((useCase, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: idx % 2 === 0 ? -20 : 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
              className="glass rounded-2xl p-6"
            >
              <h3 className="text-2xl font-semibold mb-3 gradient-text">{useCase.title}</h3>
              <p className="text-gray-300 mb-4">{useCase.description}</p>
              <div className="bg-black/30 rounded-lg p-3 text-sm text-green-400 font-mono">
                {useCase.example}
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative container mx-auto px-6 py-32">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="glass rounded-3xl p-12 text-center max-w-3xl mx-auto"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Ready to Get Started?
          </h2>
          <p className="text-xl text-gray-400 mb-4">
            Upload your first document and experience AI-powered OCR
          </p>
          <p className="text-gray-500 mb-8">
            Free tier: 1,500 pages/day • No credit card required • Process locally
          </p>
          <Link href="/upload">
            <button className="px-10 py-5 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg font-semibold text-xl hover:shadow-2xl hover:shadow-purple-500/50 transition-all duration-300">
              Start Processing Now
            </button>
          </Link>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="relative container mx-auto px-6 py-8 border-t border-white/10">
        <div className="text-center text-gray-500">
          <p>© 2025 SmartScan. Powered by Gemini AI & Tesseract OCR.</p>
          <p className="text-sm mt-2">7+ Document Types • 95%+ Accuracy • 2-4x Faster Processing</p>
        </div>
      </footer>
    </main>
  );
}
