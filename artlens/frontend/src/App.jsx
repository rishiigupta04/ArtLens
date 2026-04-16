import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import AnimatedGradientBg from './components/AnimatedGradientBg'
import CursorGlow from './components/CursorGlow'
import Home   from './pages/Home'
import Batch  from './pages/Batch'
import About  from './pages/About'

export default function App() {
  return (
    <div className="min-h-screen relative bg-transparent">
      <AnimatedGradientBg />
      <CursorGlow />
      <Navbar />
      <div className="relative z-10 pb-24 sm:pb-12">
        <Routes>
          <Route path="/"      element={<Home />} />
          <Route path="/batch" element={<Batch />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </div>
  )
}