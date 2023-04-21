import base64
import io
from matplotlib import pyplot as plt
import pandas as pd 
import dash 
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from PIL import Image, ImageDraw, ImageFont
def load_encoded_image(image_path):
    with open(image_path, 'rb') as f:
        image = f.read()
    return base64.b64encode(image).decode('ascii')

def closest_class():
    df = pd.read_csv('CSV_files\Activity_Similarity-A.csv')
    df = df[['Activity', 'Average']]
    df = df.sort_values(by=['Average']).head(5)
    x = df['Activity']
    y = df['Average']
    
    min = y.idxmin()
    max = y.idxmax()

    plt.plot(x, y, 'go')
    plt.title('Similar Activities to Walking')
    plt.text(x[min], y[min], f'#1', fontsize=12)
    plt.text(x[max], y[max], f'#5', fontsize=12)
    plt.savefig('Canvas_data\Similar_Activities.png')
    # plt.show()
    # plt.clf()

def text_to_img(file, name):
        with open(file) as f:
         text = f.read()
        img = Image.new('RGB', (300, 300), color = (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('arial.ttf', 15)
    
        draw.text((10, 10), text, font=font, fill=(0, 0, 0))
    
        img.save(f'Canvas_data/{name}.png')

text_to_img('Canvas_data/LIME-A.txt', 'LIME-Analysis')
text_to_img('Canvas_data/top_features.txt', 'top_Features')
text_to_img('Canvas_data/classification_report_A.txt', 'Classification_Report')
closest_class()





image_paths = {
    'PDP': 'Canvas_data\PDPA.png',
    'Pairwise Euclidean': 'Canvas_data\Pairwise_Euclidean-A.png',
    'Distance Measurements': 'Canvas_data\Activity_heatmap-A.png',
    'Classification Report': 'Canvas_data\classification_report_A.png',
    'Top Features': 'Canvas_data/top_features.png',
    'LIME': 'Canvas_data/LIME-Analysis.png',
    'SHAP': 'Canvas_data/SHAP-A.png',
    'Similar Activities': 'Canvas_data\Similar_Activities.png',
    
    
}
encoded_images = {label: base64.b64encode(open(image_path, 'rb').read()).decode('utf-8') for label, image_path in image_paths.items()}

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Interactive Visualization for Interpretable Machine Learning'),
    dcc.RadioItems(
        id='image-toggle',
        options=[{'label': k, 'value': k} for k in encoded_images.keys()],
        value='Classification Report',
        labelStyle={'display': 'inline-block', 'margin': '10px'}
    ),
    html.Img(id='image-display', src=f"data:image/png;base64,{encoded_images['Classification Report']}", style={'width': '50%', 'height': 'auto', 'cursor': 'pointer'})
])

@app.callback(Output('image-display', 'src'),
              Input('image-toggle', 'value'))
def update_image(value):
    return f"data:image/png;base64,{encoded_images[value]}"

if __name__ == '__main__':
   app.run_server(debug=True, host='127.0.0.1', port=8050)

