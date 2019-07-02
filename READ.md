## Klassifizierung von menschlichen Emotionen mit einem Convolutional Neural Network unter Anwendung von TensorFlow
##### Dozent: Benjamin Voigt

Hannah Rosales Musick | 0544360
Julian Höhn | 0565265
24.06.2019

---
**Beschreibung und Zweck**
In diesem Jupyter Notebook ist ein Convolutional Neural Network (CNN) implementiert, welches für die Klassifizierung von Bildern, die menschliche Emotionen zeigen, eingesetzt werden kann.

**Installations- und Nutzungshinweise**
Das Notebook wurde für den FER2013 Datensatz (_Challenges in Representation Learning: Facial Expression Recognition Challenge_, 2013) entwickelt und damit getestet. Es wird empfohlen, das Notebook auf einem Leistungsstarken (GPU-)Server laufen zu lassen. Informationen hierzu sind im Notebook zu finden.

**Theoretischer Hintergrund des Projektes**
Bereits im ersten Lebensjahr haben wir Menschen die Fähigkeit den Gemütszustand anderer Menschen schnell und unbewusst einzuschätzen und darauf zu reagieren (Leppänen & Nelson, 2008). Ob das Gegenüber nun über etwas verärgert ist oder niedergeschlagen: An der Körpersprache des Gesichtes erkennt unser Gehirn die Gefühlslage und versucht den Umständen entsprechend zu reagieren, sei es aus purem Selbsterhaltungstrieb um einer möglichen physischen Konfrontation aus dem Wege zu gehen, oder einfach um jemanden Trost zu spenden. 
Maschinen wie Computer besitzen selbst keine Emotionen und reagieren auch nicht auf die Emotionen von Menschen. Ob der eingegebene Befehl nun von einem wütenden oder traurigem Benutzer stammt, ist dem Computer gleichgültig.
Schon lange befasst sich die Wissenschaft mit der Thematik wie Computer die Emotionen von Menschen erfassen und sinnvoll verarbeiten können.
Die Entwicklung neuronaler Netzwerke hat auch in diesem Bereich geholfen, bemerkenswerte Fortschritte zu machen. Es ist nun möglich, durch Analyse einer Vielzahl an Bildern menschlicher Gesichter, Emotionen zu klassifizieren und diese in geeigneter Weise zu interpretieren.
Ziel dieses Projektes soll es sein, ein neuronales Netzwerk aufzubauen, welches anhand eines bestehenden Datensatzes von Fotografien menschlicher Gesichter, Emotionen nach Ekman (1992) klassifiziert.
Das Modell wurde mit dem FER2013 Datensatz trainiert. Der Datensatz beinhaltet 35887 Abbildungen (48x48 Pixel groß, schwarz-weiß), davon
* 4953 Wut-, 
* 547 Ekel-,
* 5121 Angst-,
* 8989 Freude-,
* 6077 Traurigkeit-,
* 4002 Überraschung- und
* 6198 Neutral-Bilder.

Achtzig Prozent der Abbildungen sind Trainingsdaten, 10 % Validierungsdaten und 10 % Testdaten.
Der Datensatz wurde im Rahmen des [“Facial Expression Recognition Challenge”](https://www.kaggle.com/c/challenges-in-representation-learningfacial-expression-recognition-challenge) auf kaggle.com erstellt. Der Gewinner konnte eine Treffgenauigkeit von 71.162% erzielen.
Im Vergleich konnten Menschen nur 65±5% erzielen (Goodfellow et al., 2013).

**Beschreibung der Implementierung**
Die Implementierung erfolgte in Python (Version 3.7.3) unter Einsatz des OpenSource Frameworks TensorFlow (Version 1.13.0). Es wurde bewusst auf den Einsatz von TensorFlow v2 verzichtet, da der benutzte Server (deepgreen03.f4.htw-berlin.de) nur die alte Version installiert hat.
Zur Versionskontrolle unserer Implementierung wurde ein [Repository mit Git](https://studi.f4.htw-berlin.de/gitweb/?p=s0544360/cnn_projekt.git) erstellt .
Das Notebook ist grob in vier Sektionen unterteilt:
* Einlesen und Vorbereiten des Datensatzes
* Modell
* Training und Evaluation des Modells
* Visualisierung

*Einlesen und Vorbereiten des Datensatzes*
Der Datensatz liegt im .csv-Format vor. Er wird eingelesen und Trainings- und Testdaten werden getrennt und für das Training vorbereitet.

*Modell*
Als Grundlage für die Wahl der für die Problemstellung am besten geeigneten Architektur, beziehen wir uns auf die Forschungsresultate von Pramerdorfer und Kampel (2016). Diese haben Architekturen im Kontext der Emotionserkennung durch ein CNN verglichen. Sie haben herausgefunden, das die drei getesteten Modelle VGG, Inception und ResNET eine sehr hohe Genauigkeit bei der Erkennung von Emotionen auf Gesichtern aufweisen und sich diese hauptsächlich durch die Anzahl verwendeter Layer und  eingesetzter Parameter unterscheiden.
Daher haben wir für unser Projekt die in dem Artikel beschriebene VGG-Architektur als Grundlage des Convolutional Neural Networks verwendet. Diese verwendet 10 Layer, hat 1.8 Millionen Parameter und die mögliche Klassifizierungsrate erreicht laut Pramerdorfer und Kampel 72.7%.

Name|Architektur|Tiefe|Parameter|Genauigkeit
--|--|--|--|--
VGG|CCPCCPCCPCCPFF|10|1.8m|72.7%

*Training des Modells*
Hier wird das Modell kompiliert. Loss-Funktion, Optimierungs-Funktion und deren Parameter werden hier festgelegt. Die Methode `fit_generator()` passt die Daten an das Modell an, dabei wird Data Augmentation verwendet.

*Visualisierung*
Zum einen wird pro Epoche ein Graph ausgegeben mit den Werten train_loss, train_acc, val_loss und val_acc. Andererseits werden zum Ende des Trainings die Loss-Werte und die Accuracy-Werte nochmal detailliert dargestellt.

**Beschreibung der Experimente und Fazit**
Es wurde versucht, das VGG Netz von Pramerdorfer und Kampel (2016) nachzubauen: Nach jedem CCP-Block und nach dem ersten Fully Connected Layer Dropout angewendet; Batch Normalization wird nach jedem Convolutional und nach jedem Fully Connected Layer hinzugefügt; Die Kernelgröße ist konstant (3x3); Es werden 300 Epochen trainiert; Als Verlustfunktion kommt Cross Entropy zum Einsatz; Die Optimierungsfunktion ist SGD. 
Aufgrund fehlender Informationen ist dies nur bedingt gelungen. Unterschiedliche Initialisierungen (der Empfehlung nach z. B. schrittweise Reduzierung der Learning Rate oder das einsetzen einer anderen Optimierungsfunktion (Adam)) haben zu keinen besseren Ergebnissen geführt.
Gründe hierfür könnten sein: 
* Dropout. Es wurde immer 0.5 als Dropout-Rate genommen. Pramendorfer und Kampel (2016) hatten zur Hyperparameteroptimierung eine Rastersuche durchgeführt, um die optimalen Dropout-Raten zu bestimmen.
* Image Preprocessing/Data Augmentation. Es wurde nur mit waagerechter Spiegelung gearbeitet.

Deshalb wurde die Architektur drastisch verkleinert. Versuche B und E sehen schon vielversprechend aus. 
Over- und Underfitting sind Probleme, die leider aufgetreten sind.
Gegen Underfitting (die Loss-Werte sind höher als die Accuracy-Werte) wurde versucht, alle Regularisierungsverfahren (BatchNorm und Dropout) rauszunehmen. Das hat dann zu Overfitting geführt, so das nach und nach wieder reguliert wurde.
Da auch bei Versuch F viel Overfitting zu beobachten ist, wurde die Anzahl der Parameter in Versuch G reduziert (von ~5 Millionen auf ~360,000) durch Reduzierung der Anzahl der Feature Maps (in Versuch G: 32 in den Conv-Layern, 128 in dem FC-Layer. Vorher 32-64 in den Conv-Layern, 1024 in dem FC-Layer).

Leider hat die Zeit gefehlt, um systematisch und umfangreich testen zu können. Keines der Ergebnisse kann als "gut" bezeichnet werden.

---

C = Convolutional Layer
P = Pooling Layer
D = Dropout (0.5)
B = Batch Normalization
F = Fully Connected Layer

|  |Architektur (+ Dropout und BN)|Optimizer|Learning Rate|Decay|Batch Size|Parameter|Ergebnis|
|--|--|--|--|--|--|--|--|
|Pramerdorfer und Kampel (2016)|CBCBPD-CBCBPD-CBCBPD-CBCBPD-FBDFB|SGD (momentum=0.9) | 0.1|0.0001 | 1,449,987 |128| ![Grafik Pramerdorfer und Kampel](https://lh3.googleusercontent.com/w7_ia4APTdXOshO0i8V5r2nHBJJjOS-yFzyEWheySBXcprx28RGgPe2Z8zxVxsB0W1B1ldwrF8eG) |
|Pramerdorfer und Kampel (2016)|CBCBPD-CBCBPD-CBCBPD-CBCBPD-FBDFB|Adam (beta1=0.9, beta2=0.999) | 0.1|0.0001 |1,449,987|128| ![Grafik Pramerdorfer und Kampel + Adam](https://lh3.googleusercontent.com/nUr_NWggTXwv-TaK8nVBBdP8ZbsQV_sXNmGV9yD2ZpxM1xS-dKIjxU-ilRlINuyfKtRAJmOSujiZ) |
|Versuch A|CBCBPD-CBCBPD-CBCBPD-CBCBD-FBDF|SGD (momentum=0.9) | 0.1|0.00001 |1,449,959|128| ![Grafik Versuch A](https://lh3.googleusercontent.com/-U9ydjTNWygcP_Vx63ePw7J2QFfg71aBUryGJvHBHkxM4ocIXYZozq14MpIzORt91swmU1LJb1u3) |
|Versuch B|CBCBPD-CBCBPD-FBDF|SGD (momentum=0.9)|0.001|-|5,386,471|128|![Grafik Versuch B](https://lh3.googleusercontent.com/t84M2Oo3n8tvvO6-RbTGM4MAdI6WZ-bJzyXRS8SUU4DgssfiE96748_1ZZHc-ZQyIiER0nCX--MD)|
|Versuch C|CBCBPD-CBCBPD-FBDFB|SGD (momentum=0.9)|0.1|0.0001|5,386,499|128|![Grafik Versuch C](https://lh3.googleusercontent.com/1zSZUZGb6LjLooHNtcwtuZDn2uG6a8EGK4Fg8y2xCJ-KRHP7lJEj1x963D8xKcqhzWSQaKH9P5Vv)|
|Versuch D|CBCBPD-CBCBPD-FBDFB|SGD (momentum=0.9)|0.001|0.0001|5,386,499|128|![Grafik Versuch D](https://lh3.googleusercontent.com/hb3jmdfpbr0Yk6ZbeMkvRIDxD_NuJRaIOP8fC976Ww6sjG7Ew1l-kpnCAnp31n5to3Q2tqaKEZGh)|
|Versuch E|CBCBP-CBCBP-FBDF|SGD (momentum=0.9)|0.1|0.0001|5,386,471|128|![Grafik Versuch E](https://lh3.googleusercontent.com/DjacwAnC2SDnDKk3LRFuTQJDHdfWsKRwFWVqazad9sJVySR7j6a-YVAgvypTI56zG15QUWHh3P1n)|
|Versuch F|CBCBPD-CBCBPD-FBDF|Adam (beta1=0.9, beta2=0.999)|0.001|0.0001|5,386,471|128|![Grafik Versuch F](https://lh3.googleusercontent.com/hC-WPnmSR1ZpS8LsnXCf03Q9OKOYRAMuLJO-RtZBosZDiGeHe-ZbOODsbo0DWaB-jCXnASH7qdbR)|
|Versuch G|CBCBPD-CBCBPD-FBDF|Adam (beta1=0.9, beta2=0.999)|0.001|0.0001|361,895|128|![Grafik Versuch G](https://lh3.googleusercontent.com/kmx6-JODz1cAjSQIKDSniRMg1neQbc4IC-eqL5ebkz6UM-yWni87aUmBSweUdXA0Ea5V6HDDjKJW)|
|Versuch H|CCP-CCP-FF|Adam (beta1=0.9, beta2=0.999)|0.001|0.0001|397,287|128| ![Graphik Versuch H](https://lh3.googleusercontent.com/kHn6icxdx_QcgDlRrgmDrjF7aTNqCb4UToUCkDoNTUzdt1-6eAZwYwSC7Kuc3VzjzcwhwXErpgdT)|


**Quellen**
- C Pramerdorfer, M Kampel. (2016). Facial Expression Recognition using Convolutional Neural Networks: State of the Art. Retrieved from:  arXiv:1612.02903 [cs.CV]
- Leppänen, Jukka M., Nelson, Charles A. (2008). Tuning the developing brain to social signals of emotions. _Nature Reviews Neuroscience 10_, 37. doi:10.1038/nrn2554
- Ekman, Paul (1992). An argument for basic emotions. _Cognition and Emotion, 6:3-4_, 169-200. doi: 10.1080/02699939208411068
- Challenges in Representation Learning: Facial Expression Recognition Challenge. (2013). Retrieved from: https://www.kaggle.com/c/challenges-in-representation-learningfacial-expression-recognition-challenge
- I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li, X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu, M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and Y. Bengio.(2013). Challenges in Representation Learning: A report on three machine 4 learning contests. Retrieved from: arXiv:1307.0414 [stat.ML]
- 

---
MIT License

Copyright (c) 2019 Hannah Rosales Musick, Julian Höhn

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is  furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

