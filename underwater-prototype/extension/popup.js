// extension/popup.js
const fileInput = document.getElementById('fileInput');
const enhanceBtn = document.getElementById('enhanceBtn');
const previewInner = document.getElementById('previewInner');
const progressArea = document.getElementById('progressArea');
const progFill = document.getElementById('progFill');
const progText = document.getElementById('progText');
const resultCard = document.getElementById('resultCard');
const metricsDiv = document.getElementById('metrics');
const downloadLink = document.getElementById('downloadLink');
const viewDash = document.getElementById('viewDashboard');

let selectedFile = null;

fileInput.addEventListener('change', (e) => {
  selectedFile = e.target.files[0];
  showPreview(selectedFile);
  resultCard.classList.add('hidden');
});

function showPreview(file) {
  previewInner.innerHTML = '';
  if(!file) { previewInner.textContent = 'No file selected'; return; }
  const url = URL.createObjectURL(file);
  if(file.type.startsWith('image/')) {
    const img = document.createElement('img'); img.src = url; previewInner.appendChild(img);
  } else if(file.type.startsWith('video/')) {
    const vid = document.createElement('video'); vid.controls = true; vid.src = url; previewInner.appendChild(vid);
  } else {
    previewInner.textContent = 'Unsupported file type';
  }
}

enhanceBtn.addEventListener('click', () => {
  if(!selectedFile) return alert('Please select a file to enhance.');
  uploadAndEnhance(selectedFile);
});

viewDash.addEventListener('click', () => {
  // open the local dashboard in a new tab
  chrome.tabs.create({ url: 'http://127.0.0.1:8000/dashboard' });
});

function uploadAndEnhance(file) {
  progressArea.classList.remove('hidden');
  progFill.style.width = '0%';
  progText.textContent = 'Uploading file…';

  const url = 'http://127.0.0.1:8000/process';

  const xhr = new XMLHttpRequest();
  xhr.open('POST', url, true);

  xhr.upload.onprogress = function(e) {
    if (e.lengthComputable) {
      const percent = Math.round((e.loaded / e.total) * 100);
      progFill.style.width = percent + '%';
      progText.textContent = `Uploading... ${percent}%`;
    }
  };

  xhr.onload = function() {
    progressArea.classList.add('hidden');
    if (xhr.status === 200) {
      try {
        const res = JSON.parse(xhr.responseText);
        if (res.status === 'ok') {
          showResult(res);
        } else {
          alert('Error: ' + (res.message || 'unknown'));
        }
      } catch (err) {
        alert('Server response parse error: ' + err);
      }
    } else {
      alert('Upload failed: ' + xhr.statusText);
    }
  };

  xhr.onerror = function() {
    progressArea.classList.add('hidden');
    alert('Network error while uploading. Is the backend running?');
  };

  const formData = new FormData();
  formData.append('file', file, file.name);
  xhr.send(formData);
}

function showResult(res) {
  resultCard.classList.remove('hidden');
  let html = '';
  if(res.metrics) {
    html += `<div><strong>PSNR:</strong> ${res.metrics.psnr ? res.metrics.psnr.toFixed(2) : 'N/A'}</div>`;
    html += `<div><strong>SSIM:</strong> ${res.metrics.ssim ? res.metrics.ssim.toFixed(3) : 'N/A'}</div>`;
    html += `<div><strong>UIQM (approx):</strong> ${res.metrics.uiqm_approx ? res.metrics.uiqm_approx.toFixed(2) : 'N/A'}</div>`;
  }
  metricsDiv.innerHTML = html;
  const fileURL = `http://127.0.0.1:8000${res.out_url}`;
  downloadLink.href = fileURL;
  downloadLink.textContent = 'Download Enhanced';
  downloadLink.target = '_blank';
}
