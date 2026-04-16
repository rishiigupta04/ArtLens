export default function About() {
  const papers = [
    {
      tag: 'CVPR 2023',
      title: 'Towards Universal Fake Image Detectors',
      author: 'Ojha et al.',
      desc: 'Justifies ViT backbone and open-set detection approach.',
      url: 'https://arxiv.org/abs/2302.10174',
    },
    {
      tag: 'CVPR 2020',
      title: 'CNN-Generated Images Are Surprisingly Easy to Spot',
      author: 'Wang et al.',
      desc: 'JPEG + blur augmentation strategy used in training.',
      url: 'https://arxiv.org/abs/1912.11035',
    },
    {
      tag: 'arXiv 2024',
      title: 'AI-Generated Image Detection: An Empirical Study',
      author: 'Tasnim et al.',
      desc: 'Evaluation protocol: AUROC + F1 + per-source accuracy.',
      url: 'https://arxiv.org/abs/2511.02791',
    },
    {
      tag: 'arXiv 2025',
      title: 'Methods and Trends in Detecting AI-Generated Images',
      author: 'Mahara & Rishe',
      desc: 'FatFormer DWT + ViT dual-branch validates Phase 4 design.',
      url: 'https://arxiv.org/abs/2502.15176',
    },
    {
      tag: 'Sci. Reports 2025',
      title: 'Detection Using Combined Uncertainty Measures',
      author: 'Anonymous',
      desc: 'Uncertainty-aware prediction and rejection mechanism.',
      url: 'https://www.nature.com/articles/s41598-025-28572-8',
    },
  ]

  const metrics = [
    ['Test accuracy',       '98.77%'],
    ['Test AUROC',          '0.9975'],
    ['Test F1',             '0.9877'],
    ['Generator accuracy',  '98.43%'],
    ['MJ accuracy',         '99.27%'],
    ['SD accuracy',         '98.43%'],
  ]

  return (
    <main className="max-w-5xl mx-auto px-4 py-6 sm:py-10 space-y-8 sm:space-y-10">
      <div>
        <h1 className="text-3xl font-display text-slate-100">About ArtLens</h1>
        <p className="text-sm text-slate-300 mt-2 leading-relaxed max-w-3xl">
          ArtLens is an open-source AI art detection system that detects AI-generated images, identifies the generator, flags unknown generators,
          and explains decisions via attention rollout and GradCAM++ heatmaps.
        </p>
      </div>

      {/* Metrics */}
      <div>
        <h2 className="text-sm font-semibold text-slate-200 mb-3">Model performance</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
          {metrics.map(([k, v]) => (
            <div key={k} className="soft-card p-4 space-y-1 interactive-lift">
              <span className="text-xs text-slate-400 block">{k}</span>
              <span className="text-lg font-semibold text-slate-100">{v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-slate-200">Architecture</h2>
        {[
          ['ViT-B/16 backbone', 'Fine-tuned on WikiArt + JourneyDB + DiffusionDB. 85.8M parameters. Multi-task: binary detection + generator fingerprinting + open-set detection.'],
          ['Frequency branch', 'FFT (18-dim) + DWT (24-dim) + LBP (28-dim) handcrafted features fed into LightGBM ensemble alongside ViT embeddings.'],
          ['Open-set detection', 'Mahalanobis distance with LedoitWolf precision matrices computed on 28,000+ training embeddings. Normalised distances flag unknown generators.'],
          ['Explainability', 'Last-layer attention rollout for global focus. GradCAM++ on blocks[-3] for decision-critical region localisation.'],
        ].map(([title, desc]) => (
          <div key={title} className="soft-card p-4 space-y-1">
            <h3 className="text-sm font-medium text-slate-100">{title}</h3>
            <p className="text-xs text-slate-300 leading-relaxed">{desc}</p>
          </div>
        ))}
      </div>

      {/* Papers */}
      <div className="space-y-3">
        <h2 className="text-sm font-semibold text-slate-200">Research papers</h2>
        {papers.map(p => (
          <a
            key={p.title}
            href={p.url}
            target="_blank"
            rel="noopener noreferrer"
            className="soft-card flex flex-col sm:flex-row gap-3 sm:gap-4 items-start p-4 interactive-lift hover:border-cyan-300/30"
          >
            <span className="text-xs bg-cyan-400/10 text-cyan-200 border border-cyan-300/30 px-2 py-0.5 rounded-full whitespace-nowrap font-medium">
              {p.tag}
            </span>
            <div>
              <p className="text-sm font-medium text-slate-100">{p.title}</p>
              <p className="text-xs text-slate-400">{p.author} · {p.desc}</p>
            </div>
          </a>
        ))}
      </div>

      <div className="text-xs text-slate-400 px-6 pt-4 border-t border-white/10 space-y-1">
        <p>Crafted with ❤️‍ by Rishi</p>
        <p>Stack: ViT-B/16 · LightGBM · FastAPI · HuggingFace Spaces · React · Vercel · Supabase</p>
      </div>
    </main>
  )
}