import grib2io
import numpy as np

grbf = grib2io.open("single_REFL_record.grib2",'r')
msg = grbf[0]
data_in = msg.data
grbf.close()

if False:
 for i in range(7):
   print()
   print("========================")
   print(f"SECTION {i} full values")
   print("========================")
   print()
   for k,v in msg.attrs_by_section(i,values=True).items():
      print(f"{k}: {v}")

 print(msg.attrs_by_section(4))
 print(msg.section4)

 sect4 = np.zeros(msg.section4.shape, dtype=msg.section4.dtype)
 print(dir(msg.attrs_by_section(4)))
 for key in msg.attrs_by_section(4):
    print(key)
    exec(f"print(msg.{key}, type(msg.{key}))")

section2 = b"Creation note: Smoothed reflectivity based on ratio between beam width and grid size"
out_msg = grib2io.Grib2Message(section0=msg.section0,section1=msg.section1,section2=section2,section3=msg.section3,section4=msg.section4,section5=msg.section5)
out_msg.data = data_in*0.001
out_msg.pack()

print(out_msg._msg[0:4],out_msg._msg[-4:])
print(out_msg)
#print(out_msg.attrs_by_section(4,values=False))

print(out_msg.shortName)
out_msg.shortName = "TMP"
print(out_msg.shortName)

grb2 = grib2io.open("new_GRIB2_file.grib2",'w')
grb2.write(out_msg)
grb2.close()
