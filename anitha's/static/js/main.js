/**
 * MediLens AI - Main Client-Side JavaScript
 * Controls drag-and-drop uploads, interactive XAI opacity sliders,
 * patch visualizer hover inspection, and animated pipeline feedback.
 */

document.addEventListener('DOMContentLoaded', () => {
    initAlertDismissal();
    initDemoLoginFiller();
    initDropzone();
    initOpacitySliders();
    initSampleSelectors();
    initAnalyzePipelineModal();
});

// Auto-dismiss alerts after 5 seconds
function initAlertDismissal() {
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            alert.style.opacity = '0';
            alert.style.transform = 'translateY(-8px)';
            setTimeout(() => alert.remove(), 500);
        }, 6000);
    });
}

// 1-Click Demo Login credentials filler
function initDemoLoginFiller() {
    const demoBtns = document.querySelectorAll('.btn-demo-creds');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const roleSelect = document.getElementById('role');

    demoBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const email = btn.dataset.email;
            const password = btn.dataset.password;
            const role = btn.dataset.role;

            if (emailInput) emailInput.value = email;
            if (passwordInput) passwordInput.value = password;
            if (roleSelect && role) roleSelect.value = role;

            // Highlight input fields
            [emailInput, passwordInput].forEach(inp => {
                if (inp) {
                    inp.style.borderColor = '#0D9488';
                    inp.style.backgroundColor = '#F0FDFA';
                    setTimeout(() => {
                        inp.style.borderColor = '';
                        inp.style.backgroundColor = '';
                    }, 1200);
                }
            });
        });
    });
}

// Drag & Drop File Upload
function initDropzone() {
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('imageFileInput');
    const previewContainer = document.getElementById('uploadPreviewContainer');
    const previewImg = document.getElementById('uploadPreviewImg');
    const fileNameSpan = document.getElementById('uploadFileName');
    const clearBtn = document.getElementById('clearUploadBtn');
    const analyzeBtn = document.getElementById('btnSubmitAnalyze');
    const sampleInput = document.getElementById('selectedSampleType');

    if (!dropzone || !fileInput) return;

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    dropzone.addEventListener('click', (e) => {
        if (e.target.closest('#clearUploadBtn') || e.target.closest('.sample-card')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileSelection(fileInput.files[0]);
        }
    });

    if (clearBtn) {
        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetUpload();
        });
    }

    function handleFileSelection(file) {
        // Validate format
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            alert('Invalid file format. Please upload a chest radiograph in JPG or PNG format.');
            return;
        }

        // Validate size (max 16 MB)
        if (file.size > 16 * 1024 * 1024) {
            alert('File exceeds the 16MB size limit.');
            return;
        }

        // Clear sample selection if any
        if (sampleInput) sampleInput.value = '';
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));

        // Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImg) previewImg.src = e.target.result;
            if (fileNameSpan) fileNameSpan.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            if (previewContainer) previewContainer.style.display = 'block';
            if (analyzeBtn) analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        if (fileInput) fileInput.value = '';
        if (sampleInput) sampleInput.value = '';
        if (previewContainer) previewContainer.style.display = 'none';
        if (previewImg) previewImg.src = '';
        if (analyzeBtn) analyzeBtn.disabled = true;
        document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
    }
}

// 1-Click Sample Chest X-Ray Selectors
function initSampleSelectors() {
    const sampleCards = document.querySelectorAll('.sample-card');
    const sampleInput = document.getElementById('selectedSampleType');
    const previewContainer = document.getElementById('uploadPreviewContainer');
    const previewImg = document.getElementById('uploadPreviewImg');
    const fileNameSpan = document.getElementById('uploadFileName');
    const analyzeBtn = document.getElementById('btnSubmitAnalyze');
    const fileInput = document.getElementById('imageFileInput');

    sampleCards.forEach(card => {
        card.addEventListener('click', () => {
            sampleCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');

            const sampleType = card.dataset.sample;
            const imgSrc = card.dataset.src;
            const label = card.dataset.label;

            if (sampleInput) sampleInput.value = sampleType;
            if (fileInput) fileInput.value = ''; // clear custom file

            if (previewImg) previewImg.src = imgSrc;
            if (fileNameSpan) fileNameSpan.textContent = `Demonstration Sample: ${label}`;
            if (previewContainer) previewContainer.style.display = 'block';
            if (analyzeBtn) analyzeBtn.disabled = false;
        });
    });
}

// Animated ViT Diagnostic Pipeline Modal
function initAnalyzePipelineModal() {
    const form = document.getElementById('uploadForm');
    const modal = document.getElementById('pipelineModal');

    if (!form || !modal) return;

    form.addEventListener('submit', () => {
        modal.classList.add('active');
        runPipelineAnimation();
    });

    function runPipelineAnimation() {
        const steps = [
            { id: 'pipeStep1', delay: 100, text: 'Resizing to 224x224 & Standardizing RGB Channels...' },
            { id: 'pipeStep2', delay: 700, text: 'Partitioning Image into 196 (16x16) Spatial Patches...' },
            { id: 'pipeStep3', delay: 1400, text: 'Projecting to 192-D Vectors & Injecting Position Embeddings...' },
            { id: 'pipeStep4', delay: 2100, text: 'Propagating through 6 Multi-Head Self-Attention Transformer Layers...' },
            { id: 'pipeStep5', delay: 2800, text: 'Computing Attention Rollout & Jet Heatmap Localization...' },
            { id: 'pipeStep6', delay: 3500, text: 'Finalizing Clinical Diagnosis & Generating Report...' }
        ];

        steps.forEach(s => {
            setTimeout(() => {
                const el = document.getElementById(s.id);
                if (el) {
                    el.classList.add('step-active');
                    const badge = el.querySelector('.step-status');
                    if (badge) badge.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing';
                }
            }, s.delay);

            setTimeout(() => {
                const el = document.getElementById(s.id);
                if (el) {
                    el.classList.remove('step-active');
                    el.classList.add('step-done');
                    const badge = el.querySelector('.step-status');
                    if (badge) badge.innerHTML = '<i class="fas fa-check-circle" style="color: #10B981;"></i> Complete';
                }
            }, s.delay + 600);
        });
    }
}

// Interactive Heatmap Opacity Slider
function initOpacitySliders() {
    const slider = document.getElementById('heatmapOpacitySlider');
    const overlay = document.getElementById('interactiveOverlayImg');
    const opacityLabel = document.getElementById('opacityValueLabel');

    if (slider && overlay) {
        slider.addEventListener('input', (e) => {
            const val = e.target.value;
            overlay.style.opacity = (val / 100).toString();
            if (opacityLabel) opacityLabel.textContent = `${val}%`;
        });
    }
}

// Modal open/close helpers
function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.add('active');
}

function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.remove('active');
}
