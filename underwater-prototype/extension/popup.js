// extension/popup.js
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const openDashboard = document.getElementById('openDashboard');
const status = document.getElementById('status');
const orig = document.getElementById('orig');
const annot = document.getElementById('annot');
const tableWrap = document.getElementById('tableWrap');
const detTableBody = document.querySelector('#detTable tbody');
const alertBox = document.getElementById('alertBox');
const downloadEnhanced = document.getElementById('downloadEnhanced');
const downloadAnnotated = document.getElementById('downloadAnnotated');

let selectedFile = null;

fileInput.addEventListener('change', e => {
  selectedFile = e.target.files[0];
  status.textContent = selectedFile ? `Selected: ${selectedFile.name}` : 'No file selected';
  tableWrap.classList.add('hidden');
  // preview original
  orig.innerHTML = '';
  const url = URL.createObjectURL(selectedFile);
  if (selectedFile && selectedFile.type.startsWith('image/')) {
    const img = document.createElement('img'); img.src = url; img.style.maxWidth='100%'; orig.appendChild(img);
  } else {
    orig.innerHTML = '<div style="color:#98a8bf">Video selected (preview not available)</div>';
  }
});

openDashboard.addEventListener('click', () => {
  // open dashboard page (same backend)
  chrome.tabs.create({ url: 'http://127.0.0.1:8000/dashboard' });
});

processBtn.addEventListener('click', async () => {
  if (!selectedFile) return alert('Please select an image or video file.');
  status.textContent = 'Uploading & processing...';
  const form = new FormData();
  form.append('file', selectedFile, selectedFile.name);
  form.append('conf_threshold', 0.25);
  try {
    const res = await fetch('http://127.0.0.1:8000/process', { method: 'POST', body: form });
    const j = await res.json();
    if (j.status !== 'ok') {
      status.textContent = 'Error: ' + (j.message || 'Processing failed');
      return;
    }
    status.textContent = `Done: ${j.filename || j.filename}`;
    // show annotated preview (if image)
    tableWrap.classList.remove('hidden');
    alertBox.textContent = j.alert || '';
    detTableBody.innerHTML = '';
    (j.detections || []).forEach((d, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${idx+1}</td><td>${d.label || d.raw_label || d.class}</td><td>${(d.conf||d.confidence||0).toFixed(2)}</td><td>${d.bbox? d.bbox.join(', '): ''}</td>`;
      detTableBody.appendChild(tr);
    });
    // set download links
    downloadEnhanced.href = j.enhanced ? `http://127.0.0.1:8000/download/${j.enhanced}` : '#';
    downloadAnnotated.href = j.detected ? `http://127.0.0.1:8000/download/${j.detected}` : (j.annotated_path? j.annotated_path : '#');
    // display annotated image if available and file is image
    if (selectedFile.type.startsWith('image/')) {
      annot.innerHTML = '';
      const img = document.createElement('img');
      // annotated path: j.detected
      const annName = j.detected || j.annotated_path || '';
      if (annName) {
        img.src = `http://127.0.0.1:8000/download/${annName}`;
        img.style.maxWidth='100%';
        annot.appendChild(img);
      } else {
        annot.textContent = 'Annotated image not available';
      }
    } else {
      annot.innerHTML = '<div style="color:#98a8bf">Video processed — download annotated video below</div>';
    }
  } catch (err) {
    status.textContent = 'Network or server error: ' + err;
  }
});
