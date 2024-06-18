document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const uploadButton = document.getElementById('uploadButton');
    const annotationCanvas = document.getElementById('annotationCanvas');
    const ctx = annotationCanvas.getContext('2d');
    const labelInput = document.getElementById('labelInput');
    const saveButton = document.getElementById('saveButton');
    const downloadButton = document.getElementById('downloadButton');

    let annotations = [];
    let isDrawing = false;
    let startX, startY;

    uploadButton.addEventListener('click', () => {
        const file = imageUpload.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    annotationCanvas.width = img.width;
                    annotationCanvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    annotationCanvas.addEventListener('mousedown', (e) => {
        startX = e.offsetX;
        startY = e.offsetY;
        isDrawing = true;
    });

    annotationCanvas.addEventListener('mousemove', (e) => {
        if (isDrawing) {
            const mouseX = e.offsetX;
            const mouseY = e.offsetY;
            ctx.clearRect(0, 0, annotationCanvas.width, annotationCanvas.height);
            ctx.strokeRect(startX, startY, mouseX - startX, mouseY - startY);
        }
    });

    annotationCanvas.addEventListener('mouseup', (e) => {
        if (isDrawing) {
            isDrawing = false;
            const mouseX = e.offsetX;
            const mouseY = e.offsetY;
            const width = mouseX - startX;
            const height = mouseY - startY;
            const label = labelInput.value;
            if (label) {
                annotations.push({ x: startX, y: startY, width, height, label });
                ctx.strokeRect(startX, startY, width, height);
                ctx.fillText(label, startX, startY - 5);
            } else {
                alert('Please enter a label.');
            }
        }
    });

    saveButton.addEventListener('click', () => {
        const label = labelInput.value;
        if (label) {
            localStorage.setItem('annotations', JSON.stringify(annotations));
            alert('Annotations saved.');
        } else {
            alert('Please enter a label.');
        }
    });

    downloadButton.addEventListener('click', () => {
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(annotations));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "annotations.json");
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    });
});
