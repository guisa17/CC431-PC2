import glob
import tempfile
import os

from PIL import Image
from flask import Flask, request, redirect, send_file
from skimage import io
import base64
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

main_html = """
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" type="image/png" sizes="32x32" href="favicon-32x32.png">
    <link rel="icon" type="image/png" sizes="16x16" href="favicon-16x16.png">
    <title>Recoleccion</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">

    <style>
      #resultado {
        font-weight:  bold;
        font-size:  6rem;
        text-align: center;
      }

      .canvas-container {
          margin: 0 auto;
          border: 1px solid #ccc;
      }
    </style>

</head>
<script>
  var mousePressed = false;
  var lastX, lastY;
  var ctx;

   function getRndInteger(min, max) {
    return Math.floor(Math.random() * (max - min) ) + min;
   }

  function InitThis() {
      ctx = document.getElementById('myCanvas').getContext("2d");
      var randnumber = getRndInteger(0, 3);
      var objetos = ["fish", "car", "tree"];
      var aleatorio = objetos[randnumber];
      document.getElementById('mensaje').innerHTML  = 'Dibuje un ' + aleatorio;
      document.getElementById('objeto').value = aleatorio;

      $('#myCanvas').mousedown(function (e) {
          mousePressed = true;
          Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
      });

      $('#myCanvas').mousemove(function (e) {
          if (mousePressed) {
              Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
          }
      });

      $('#myCanvas').mouseup(function (e) {
          mousePressed = false;
      });
      
  	  $('#myCanvas').mouseleave(function (e) {
          mousePressed = false;
      });
  }

  function Draw(x, y, isDown) {
      if (isDown) {
          ctx.beginPath();
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 11;
          ctx.lineJoin = "round";
          ctx.moveTo(lastX, lastY);
          ctx.lineTo(x, y);
          ctx.closePath();
          ctx.stroke();
      }
      lastX = x; lastY = y;
  }

  function clearArea() {
      // Use the identity matrix while clearing the canvas
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  //https://www.askingbox.com/tutorial/send-html5-canvas-as-image-to-server
  function prepareImg() {
     var canvas = document.getElementById('myCanvas');
     document.getElementById('myImage').value = canvas.toDataURL();
  }

</script>

<body onload="InitThis();">
	<header>
	<div class="px-4 py-2 my-2 text-center border-bottom">
		<img class="d-block mx-auto mb-2" src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Uni-logo_transparente_granate.png/477px-Uni-logo_transparente_granate.png" alt="logo-uni" width="120" height="160">
		<h1 class="display-5 fw-bold">Recoleccion de Imagenes</h1>
		<div class="col-lg-6 mx-auto">
		</div>
      </div>
      </header>
      <main>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
    <script type="text/javascript" ></script>

    <div class="text-center">
    <h1 id="mensaje">Dibujando...</h1>
    <canvas id="myCanvas" width="200" height="200" style="border:2px solid black"></canvas>
    <br/><br/>
    <button class="btn btn-primary mx-2" onclick="javascript:clearArea();return false;">Borrar</button>
    <form class="d-inline" method="post" action="upload" onsubmit="javascript:prepareImg();" enctype="multipart/form-data">
        <input id="objeto" name="objeto" type="hidden" value="">
        <input id="myImage" name="myImage" type="hidden" value="">
        <input class="btn btn-success mx-2" id="bt_upload" type="submit" value="Enviar">
    </form>
</div>

	</main>

    
    <footer>
    <div class="b-example-divider"></div>

      <div class="bg-dark text-secondary mt-5 px-4 py-2 text-center">
        <div class="py-5">
          <h1 class="display-5 fw-bold text-white">Computacion Grafica</h1>
          <div class="col-lg-6 mx-auto">
            <p class="display-6 mb-4">CC431 Seccion A - 03/05/2024</p>
            <ul class="list-unstyled text-white">
                <li>by Espinoza Pari Franklin</li>
                <li>by Guillermo Ronie Salcedo Alvarez</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="b-example-divider mb-0"></div>
     </footer>
</body>
</html>


"""

@app.route("/")
def main():
    return(main_html)


@app.route('/upload', methods=['POST'])
def upload():
    try:
        img_data = request.form.get('myImage').replace("data:image/png;base64,", "")
        objeto = request.form.get('objeto')
        with tempfile.NamedTemporaryFile(delete=False, mode="w+b", suffix='.png', dir=str(objeto)) as fh:
            fh.write(base64.b64decode(img_data))
        
        print("Image uploaded")
        # ======================================================================
        # Cargar la imagen, redimensionarla a (28, 28) y convertirla a escala de grises
        image = Image.open(fh.name)
        image = image.resize((28, 28))
        image_array = np.array(image) 
        image_array = image_array[:, :, 3]
        # Normalizamos
        image_array = image_array / 255.0 
        # Redimensionamos
        image_array = np.expand_dims(image_array, axis=0)

        return main_html
        

    except Exception as err:
        print("Error occurred")
        print(err)

    return redirect("/", code=302)


@app.route('/prepare', methods=['GET'])
def prepare_dataset():
    images = []
    objetos = ['fish', 'car', 'tree']
    misObjetos = []
    for objeto in objetos:
        filelist = glob.glob('{}/*.png'.format(objeto))
        images_read = io.concatenate_images(io.imread_collection(filelist))
        images_read = images_read[:, :, :, 3]
        objetos_read = np.array([objeto] * images_read.shape[0])
        images.append(images_read)
        misObjetos.append(objetos_read)
    images = np.vstack(images)
    misObjetos = np.concatenate(misObjetos)
    np.save('X.npy', images)
    np.save('y.npy', misObjetos)
    return "OK!"

@app.route('/X.npy', methods=['GET'])
def download_X():
    return send_file('./X.npy')

@app.route('/y.npy', methods=['GET'])
def download_y():
    return send_file('./y.npy')

if __name__ == "__main__":
    objetos = ['fish', 'car', 'tree']
    for objeto in objetos:
        if not os.path.exists(str(objeto)):
            os.mkdir(str(objeto))
    app.run()