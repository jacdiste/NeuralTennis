<h1 align="center">NEURAL TENNIS</h1>

<p align="center">
  <img src="media\logoscritta.png"/>
</p>

<p align="center">NeuralTennis è un programma di detection e tracking di giocatori e pallina in un match di tennis tramite l'utilizzo di YOLO. L'applicazione si occupa anche di estrarre i keypoints del campo grazie alla rete neurale ResNet50, modificata appositamente per questo compito.</p>

<br><br>

<h2 align="center">IMPLEMENTAZIONE</h2>

<p align="center">Per il funzionamento del programma sono stati utilizzati:</p>

<ul align="center">
  <li>YOLOv8 per la detection e il tracking dei giocatori</li>
  <li>YOLOv5 fine-tuned su un dataset di match di tennis per la detection della pallina</li>
  <li>Interpolazione per il tracking della pallina</li>
  <li>ResNet50 rimodulata in head con training su dataset specifico per l'estrazione dei keypoints del campo</li>
</ul>

<br>

<h2 align="center">OUTPUT</h2>
<p align="center">Segue un output generato dal programma che mostra il tracking di giocatori e pallina tramite bounding boxes, e i 14 keypoints estratti dalle intersezioni delle linee di campo:</p>

<p align="center">
  <img src="media\2_out_gif.gif"/>
</p>

<br><br>

<p align="center">Autore: Di Stefano Jacopo (1996080)</p>

<br>

<p align="center">
  <img src="media\logo.png" width=100/>
</p>