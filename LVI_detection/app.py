from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory, Response, send_file
import os
import uuid
import threading
import subprocess
import time
import json
import zipfile
from io import BytesIO
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(BASE_DIR, 'results')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'ndpi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_ndpi_file(ndpi_path, job_id):
    try:
        result_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
        os.makedirs(result_dir, exist_ok=True)

        def write_status(stage):
            with open(os.path.join(result_dir, "status.txt"), "w") as f:
                f.write(stage)
        
        write_status("vascular_running")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Starting full NDPI analysis pipeline...")

        ndpi_abs_path = os.path.abspath(ndpi_path)
        vascular_candidates_dir = os.path.join(result_dir, "vascular_candidates")
        os.makedirs(vascular_candidates_dir, exist_ok=True)

        # === 1. Vascular Detection ===
        vascular_script = os.path.join(BASE_DIR, "vascular_detection_model", "main.py")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running Vascular Detection...")
        subprocess.run([
            "python", vascular_script,
            "--ndpi_path", ndpi_abs_path,
            "--output", result_dir,
            "--lvi-dir", vascular_candidates_dir
        ], check=True, cwd=BASE_DIR)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Vascular detection completed.")
        write_status("lvi_running")

        # === 2. LVI Judgment ===
        lvi_judgment_script = os.path.join(BASE_DIR, "LVI_detection_model", "judgment.py")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running LVI Judgment...")
        subprocess.run([
            "python", lvi_judgment_script,
            "--scan_folder", result_dir,
        ], cwd=result_dir, check=True)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ LVI judgment completed.")
        write_status("analysis_running")

        # === 3. Check Result & Run Transfer if needed ===
        lvi_result_json = os.path.join(result_dir, "lvi_result.json")
        is_lvi_positive = False

        if os.path.exists(lvi_result_json):
            with open(lvi_result_json, "r") as f:
                data = json.load(f)
                is_lvi_positive = data.get("slide_is_lvi_positive", False)

            if is_lvi_positive:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ LVI Positive → Running Transfer Inference...")
                transfer_script = os.path.join(BASE_DIR, "LVI_detection_model", "transfer", "infer_lvi.py")
                vascular_csv_path = os.path.join(result_dir, "lvi_candidates_with_conch_LVI.csv")

                subprocess.run([
                    "python", transfer_script,
                    "--ndpi_path", ndpi_abs_path,
                    "--vascular_csv", vascular_csv_path,
                    "--output", result_dir
                ], check=True, cwd=BASE_DIR)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✅ Transfer inference completed.")
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⏭️ LVI Negative → Skipping transfer.")
        
        write_status("completed")

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🎉 All processing finished!")

    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ ERROR: {str(e)}")
        with open(os.path.join(result_dir, "status.txt"), "w") as f:
            f.write(f"failed: {str(e)}")

@app.route('/logs/<job_id>')
def stream_logs(job_id):
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    
    def generate():
        last_size = 0
        log_file = os.path.join(result_dir, "log.txt")
        
        try:
            while True:
                if os.path.exists(log_file):
                    try:
                        with open(log_file, "r", encoding="utf-8") as f:
                            f.seek(last_size)
                            new_content = f.read()
                            if new_content:
                                for line in new_content.splitlines():
                                    if line.strip():
                                        yield f"data: {line}\n\n"
                                last_size = f.tell()
                    except Exception:
                        pass

                yield "data: \n\n"  # heartbeat
                time.sleep(1.2)

        except GeneratorExit:
            pass
        except Exception as e:
            print(f"[SSE Stream Error] {e}")
    
    return Response(generate(), mimetype='text/event-stream')


# ====================== ROUTES ======================
@app.route('/')
def index():
    return redirect(url_for('upload'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Please upload a valid .ndpi file"}), 400

        job_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        ndpi_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")

        file.save(ndpi_path)

        # Save file info for review page
        file_info = {
            "name": file.filename,
            "size": os.path.getsize(ndpi_path),
            "lastModified": int(time.time() * 1000)
        }
        # Store in session (for this request)
        # We'll pass it via template in review

        thread = threading.Thread(target=process_ndpi_file, args=(ndpi_path, job_id))
        thread.daemon = True
        thread.start()

        return redirect(url_for('review', job_id=job_id))

    return render_template('upload.html')

@app.route('/review/<job_id>')
def review(job_id):
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    status_file = os.path.join(result_dir, "status.txt")

    status = "processing"
    vascular_heatmap = None
    lvi_heatmap = None
    is_lvi_positive = False

    if os.path.exists(status_file):
        with open(status_file) as f:
            content = f.read().strip()
            if content.startswith("completed"):
                status = "completed"
            elif content.startswith("failed"):
                status = "failed"

    # If completed → redirect to final preview page
    if status == "completed":
        # Find heatmaps (same logic you already have)
        for f in os.listdir(result_dir):
            if not f.endswith(('.png', '.jpg', '.jpeg')): continue
            lower = f.lower()
            if ("vascular_thresholded_heatmap" in lower):
                vascular_heatmap = f"/results/{job_id}/{f}"
            if "lvi_thresholded_heatmap" in lower:
                lvi_heatmap = f"/results/{job_id}/{f}"

        lvi_json = os.path.join(result_dir, "lvi_result.json")
        if os.path.exists(lvi_json):
            try:
                with open(lvi_json) as f:
                    data = json.load(f)
                    is_lvi_positive = data.get("slide_is_lvi_positive", False)
            except:
                pass

        return render_template('preview.html',
                               job_id=job_id,
                               vascular_heatmap=vascular_heatmap,
                               lvi_heatmap=lvi_heatmap,
                               is_lvi_positive=is_lvi_positive)

    # Still processing → show live review page
    return render_template('review.html',
                           job_id=job_id,
                           status=status,
                           vascular_heatmap=None,
                           lvi_heatmap=None,
                           is_lvi_positive=False)

@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/status/<job_id>')
def get_status(job_id):
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    status_file = os.path.join(result_dir, "status.txt")
    
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            status = f.read().strip()
        return jsonify({"status": status})
    return jsonify({"status": "processing"})

@app.route('/download_results/<job_id>')
def download_results(job_id):
    result_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    
    if not os.path.exists(result_dir):
        return "Result folder not found", 404

    try:
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            files_added = 0
            
            for filename in os.listdir(result_dir):
                file_path = os.path.join(result_dir, filename)
                
                # Add heatmaps
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    zf.write(file_path, f"heatmaps/{filename}")
                    files_added += 1
                    print(f"Added to zip: {filename}")
                
                # Add CSV files
                elif filename.lower().endswith('.csv'):
                    zf.write(file_path, f"data/{filename}")
                    files_added += 1
                    print(f"Added to zip: {filename}")
                
                # Add JSON result
                elif filename == "lvi_result.json":
                    zf.write(file_path, f"data/{filename}")
                    files_added += 1

            if files_added == 0:
                return "No result files found to download", 404

        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f"NDPI_Results_{job_id[:8]}.zip"
        )
        
    except Exception as e:
        print(f"Download error for job {job_id}: {str(e)}")
        return f"Download failed: {str(e)}", 500

if __name__ == '__main__':
    print("Flask app started at http://127.0.0.1:5000")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)