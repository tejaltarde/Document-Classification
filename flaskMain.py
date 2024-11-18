import os
from flask import Flask, request, render_template, redirect, url_for
from main import DocumentClassification

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

classifier = DocumentClassification(data_directory='dataset/bbc/')

# Train the Naive Bayes classifier on the training dataset
classifier.train_naive_bayes_classifier()

@app.route('/', methods=['GET', 'POST'])
def index():
    classification = None
    original_text = None
    preprocessed_text = None
    confusion_matrix = None
    accuracy = None
    graph = None
    error = None

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'pdf_file' not in request.files:
            error = 'No file was uploaded.'
        else:
            pdf_file = request.files['pdf_file']
            # Check if the file is a PDF
            if pdf_file.filename.endswith('.pdf'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
                pdf_file.save(file_path)
                # Classify the PDF file using the trained model
                original_text,preprocessed_text,classification = classifier.classify_pdf_file(file_path)
                # Evaluate the model on the test dataset and generate the confusion matrix and accuracy
                confusion_matrix, accuracy = classifier.evaluate_model()
                # Generate the accuracy graph
                # labels = sorted(list(set(confusion_matrix.flatten())))
                # graph_path = os.path.join('static', 'graph.png')
                # classifier.plot_confusion_matrix(confusion_matrix, labels, graph_path)
                # graph = '/' + graph_path
                return redirect(url_for('result_page', classification=classification,original_text=original_text,preprocessed_text=preprocessed_text))
            else:
                error = 'The uploaded file is not a PDF.'

    return render_template('index.html', classification=classification,original_text=original_text,preprocessed_text=preprocessed_text,
                           accuracy=accuracy)
@app.route('/result_page')
def result_page():
    classification = request.args.get('classification')
    original_text = request.args.get('original_text')
    preprocessed_text = request.args.get('preprocessed_text')
    return render_template('temp.html', classification=classification,original_text=original_text,preprocessed_text=preprocessed_text)
 
if __name__ == '__main__':
    app.run(debug=True)
