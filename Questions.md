### En fråga om dimensionerna

Frågan gäller följade kodstycke

```python
sub_images_coarse = tf.constant(value = np.moveaxis(sub_images[0:223, 0:303, :, :], -1, 0), dtype = tf.float32, name = "images_coarse") 
sub_images_fine = tf.constant(value = np.moveaxis(sub_images[0:227, 0:303, :, :], -1, 0), dtype = tf.float32, name = "images_fine") 
depthmaps_groundtruth = tf.constant(value = np.moveaxis(sub_depths[0:55, 0:74, :], -1, 0), dtype = tf.float32, name = "depthmaps_groundtruth")
```
Varför görs np.moveaxis här? För om jag har fattat det rätt så kommer inputen in med dimensionerna [15, 480, 640, 3] och sedan in i nätverken med dimensionerna (för fine) [15, 227, 303, 3], placholdern har den den dimensionen. Men om vi flyttar axlarna med np.moveaxis får vi väl dimensionen [3, 15, 303, 604], eller det är i alla fall vad mina tester i python leder till. När jag också försöker köra koden leder det till att jag får ett dimensionsfel.

Så de slutigiltiga frågorna är:

- Vilken form vill vi ha dem på? [15, 227, 303, 3] eller [3, 15, 303, 604]? Jag kan gärna ändra så att det funkar men vill bara veta vilket sätt ni hade tänkte er på. Kan också vara jag som har missuppfattat något.

- Vill vi göra så att vi bara väljer ut de 227 X 303 första pixlarna eller är det något vi bara gör så länge för att vi inte har resizat datan än? 


### Svar

np.moveaxis ändrar bilden från shape (height, width, batch_size, channels) till (batch_size, width, height, channels). network_functions_2 är skriven så att den hanterar 4d vektorer med shape (batch_size, width, height, channels), vilket är den önskvärda frmen eftersom tensorflow internt hanterar sådana vektorer.

Eftersom read_data kan ha ändrats till att justera bilderna och flytta runt axlar vid det här laget så kan det vara så att dessa transformationer är onödiga att göra i depth_prediction_2. Jag tycker att vi bör undvika så långt som möjligt att justera bilderna i filen depth_prediction och gör det istället i read_data. Ni får gärna justera filen så att den fungerar med er nya kod!  

