// extension/popup.js
const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const openDashboard = document.getElementById('openDashboard');
const status = document.getElementById('status');
const orig = document.getElementById('orig');
const enhanced = document.getElementById('enhanced');
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
  enhanced.innerHTML = '';
  annot.innerHTML = '';
  const url = URL.createObjectURL(selectedFile);
  if (selectedFile && selectedFile.type.startsWith('image/')) {
    const img = document.createElement('img');
    img.src = url;
    img.style.maxWidth = '100%';
    orig.appendChild(img);
  } else {
    orig.innerHTML = '<div style="color:#98a8bf">Video selected (preview not available)</div>';
  }
});

openDashboard.addEventListener('click', () => {
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
    status.textContent = `Done: ${j.filename || selectedFile.name}`;

    tableWrap.classList.remove('hidden');
    alertBox.textContent = j.alert || '';
    detTableBody.innerHTML = '';

    (j.detections || []).forEach((d, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${idx+1}</td><td>${d.label || d.raw_label || d.class}</td><td>${(d.conf||d.confidence||0).toFixed(2)}</td><td>${d.bbox? d.bbox.join(', '): ''}</td>`;
      detTableBody.appendChild(tr);
    });

    // set download links
    downloadEnhanced.href = j.enhanced_path ? `http://127.0.0.1:8000/download/${j.enhanced_path.split('/').pop()}` : '#';
    downloadAnnotated.href = j.annotated_path ? `http://127.0.0.1:8000/download/${j.annotated_path.split('/').pop()}` : '#';

    // display enhanced & annotated images for image files
    if (selectedFile.type.startsWith('image/')) {
      enhanced.innerHTML = '';
      annot.innerHTML = '';

      // Enhanced image
      const enhImg = document.createElement('img');
      if (j.enhanced_path) {
        enhImg.src = `http://127.0.0.1:8000/download/${j.enhanced_path.split('/').pop()}`;
        enhImg.style.maxWidth = '100%';
        enhanced.appendChild(enhImg);
      } else {
        enhanced.textContent = 'Enhanced image not available';
      }

      // Annotated image
      const annImg = document.createElement('img');
      if (j.annotated_path) {
        annImg.src = `http://127.0.0.1:8000/download/${j.annotated_path.split('/').pop()}`;
        annImg.style.maxWidth = '100%';
        annot.appendChild(annImg);
      } else {
        annot.textContent = 'Annotated image not available';
      }
    } else {
      enhanced.innerHTML = '<div style="color:#98a8bf">Video processed — download enhanced video below</div>';
      annot.innerHTML = '<div style="color:#98a8bf">Video processed — download annotated video below</div>';
    }

  } catch (err) {
    status.textContent = 'Network or server error: ' + err;
  }
});
