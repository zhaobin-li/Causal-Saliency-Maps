import yappi

def a():
    for _ in range(10000000):  # do something CPU heavy
        pass

yappi.set_clock_type("cpu") # Use set_clock_type("wall") for wall time
yappi.start()
a()

yappi.get_func_stats().print_all()
yappi.get_thread_stats().print_all()
'''

Clock type: CPU
Ordered by: totaltime, desc

name                                  ncall  tsub      ttot      tavg      
doc.py:5 a                            1      0.117907  0.117907  0.117907

name           id     tid              ttot      scnt        
_MainThread    0      139867147315008  0.118297  1
'''