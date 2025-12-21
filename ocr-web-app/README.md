# SmartScan - AI-Powered OCR Web Application

Beautiful, premium web application for OCR processing with Gemini AI enhancement.

## 🚀 Quick Start

### Backend Setup

1. Navigate to backend directory:
```bash
cd ocr-web-app/backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd ocr-web-app/frontend
```

2. Install dependencies (already done during setup):
```bash
npm install
```

3. Create `.env.local` file with:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

4. Run the development server:
```bash
npm run dev
```

The web app will be available at `http://localhost:3000`

## 📁 Project Structure

```
ocr-web-app/
├── backend/                  # FastAPI server
│   ├── main.py              # API endpoints
│   ├── requirements.txt     # Python dependencies
│   └── README.md
└── frontend/                # Next.js app
    ├── app/
    │   ├── page.tsx        # Landing page
    │   ├── upload/page.tsx # Upload interface
    │   ├── results/page.tsx # Results viewer
    │   └── globals.css     # Styles
    ├── package.json
    └── tailwind.config.ts
```

## 🎨 Features

**Landing Page:**
- Hero section with gradient animations
- Feature highlights
- Use case examples
- Premium dark mode design

**Upload Page:**
- Drag-and-drop file upload
- Multi-file support (PDF, PNG, JPG, JPEG)
- File preview and management
- Real-time validation

**Results Page:**
- Real-time processing status
- Progress tracking
- Downloadable JSON results
- Text preview

## 🛠️ Tech Stack

- **Frontend:** Next.js 14, TypeScript, Tailwind CSS, Framer Motion
- **Backend:** FastAPI, Python
- **Icons:** Lucide React
- **File Upload:** react-dropzone
- **HTTP Client:** axios

## 📝 API Endpoints

- `POST /api/ocr/upload` - Upload files for processing
- `GET /api/ocr/status/{job_id}` - Get processing status
- `GET /api/ocr/results/{job_id}` - Get OCR results

## 🎯 Usage

1. Open `http://localhost:3000`
2. Click "Start Scanning" or navigate to Upload page
3. Drag & drop or click to select files
4. Click "Start Processing"
5. Wait for processing to complete
6. Download results as JSON

## 🔧 Development

**Frontend:**
```bash
cd frontend
npm run dev      # Start dev server
npm run build    # Build for production
npm run start    # Start production server
```

**Backend:**
```bash
cd backend
python main.py   # Start API server (auto-reload enabled)
```

## 🌟 Next Steps

- Integrate actual OCR processing logic from `main.py`
- Add user authentication
- Implement result history
- Add batch processing
- Deploy to production

## 📄 License

Same as parent OCR project.
