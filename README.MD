#### Code used in protection engineering
This repository is a place where I try to add some of my code that I use in my work.

At the moment the code is a bit bad structured, but eventually it will become better.

To use some of the code in your online notebook run:
```python
!curl https://raw.githubusercontent.com/smileynor/Protection/master/distance.py -o ../distance.py

import distance as prot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example impedansediagram
a = prot.impedanceDiagram("U1")
a.add_first_section(4.0,4.0,"A")
a.add_second_section(3.0,3.0,"A","B")
a.add_third_section(4.0,4.0,"B","C")
a.add_third_section(5.0,3.0,"B","D")
a.draw_setup()
a.draw_first_section()
a.draw_second_section()
a.draw_third_section()
b = prot.siprotec_7ST63x_settings(5,8,13)
b.draw_all(a.ax)
a.draw_load_at_beginning()

for r,x,i in a.first_sections:
    a.draw_load_at_end(r,x)
    a.draw_min_SSC_at_beginning(4)
    a.draw_min_SSC_at_section_end(4,r,x)
    a.draw_infeed(4,4,b.paths[1])
    
a.ax.set_xlim(-15,20)
a.ax.set_ylim(-5,20)
plt.show()
```


__TODO__
* Add a report generator, showing the result in a nice way
* Adding some more devices
* 