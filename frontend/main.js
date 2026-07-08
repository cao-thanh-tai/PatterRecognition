const API_BASE_URL = "http://127.0.0.1:8000";

const imageInput = document.getElementById("imageInput");
const imageBtn = document.getElementById("imageBtn");
const videoBtn = document.getElementById("videoBtn");
const resultBox = document.getElementById("resultBox");
const statusBadge = document.getElementById("statusBadge");

function setStatus(type, text) {
	statusBadge.className = `status ${type}`;
	statusBadge.textContent = text;
}

function setResult(value) {
	resultBox.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

async function postJson(url, body) {
	const response = await fetch(url, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	});

	const payload = await response.json().catch(() => ({}));
	if (!response.ok) {
		const message = payload.detail || payload.message || `Request failed with status ${response.status}`;
		throw new Error(message);
	}

	return payload;
}

async function uploadImage() {
	const file = imageInput.files?.[0];
	if (!file) {
		setStatus("error", "Chưa chọn ảnh");
		setResult("Vui lòng chọn một file ảnh trước khi gửi.");
		return;
	}

	const formData = new FormData();
	formData.append("uploaded_image", file);
    console.log(formData.get("uploaded_image"));
	imageBtn.disabled = true;
	setStatus("loading", "Đang xử lý ảnh...");
	setResult("Đang gửi ảnh lên API...");

	try {
		const response = await fetch(`${API_BASE_URL}/predict/image`, {
			method: "POST",
			body: formData,
		});

		const payload = await response.json();
		if (!response.ok) {
			throw new Error(payload.detail || "Không thể xử lý ảnh.");
		}

		setStatus("success", "Xử lý xong");
		setResult(payload.results);
	} catch (error) {
		setStatus("error", "Lỗi xử lý ảnh");
		setResult(error.message);
	} finally {
		imageBtn.disabled = false;
	}
}

async function runVideoToText() {
	videoBtn.disabled = true;
	setStatus("loading", "Đang mở video...");
	setResult("Đang gọi API video. Nếu backend mở webcam, trình duyệt sẽ chờ đến khi luồng kết thúc.");

	try {
		const payload = await postJson(`${API_BASE_URL}/predict/video`, {});
		setStatus("success", "Xử lý xong");
		setResult(payload.results);
	} catch (error) {
		setStatus("error", "Lỗi xử lý video");
		setResult(error.message);
	} finally {
		videoBtn.disabled = false;
	}
}

imageBtn.addEventListener("click", uploadImage);
videoBtn.addEventListener("click", runVideoToText);
