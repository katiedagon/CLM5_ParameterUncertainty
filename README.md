README

1/30/18
Attempting to install pyDOE package in order to use lhs function
https://pythonhosted.org/pyDOE/randomized.html#latin-hypercube

First create a clone of NCAR package library
https://www2.cisl.ucar.edu/resources/computational-systems/cheyenne/software/python
>ncar_pylib -c 20180129 /glade/p/work/kdagon/ncar_pylib_clone
>cd /glade/p/work/kdagon/ncar_pylib_clone

Edit /bin/activate to change VIRTUAL_ENV variable to be /glade/p/work/kdagon/ncar_pylib_clone

Then source the virtual environment
>source /bin/activate

Install package with specified install dir
>pip install --upgrade -t /glade/p/work/kdagon/ncar_pylib_clone/lib/python3.6/site-packages pyDOE

Import pyDOE in python
>python
>import pyDOE
>help(pyDOE.lhs)
