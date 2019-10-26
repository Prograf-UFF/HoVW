#!/usr/bin/env python

import os, time

class Log(object):
    """General logs files management.

    Parameters
    ----------
    path: Path for log's file.
    name: Log's file name.

    Attributes
    ----------
    path: string
        Path where the log file is in.
    name: string, default = None
        Name of the log file.
    stime: time
        Object instantiation time.
    etime: time, default = None
        File closing time.
    log_file: file
        File object.
    """

    def __init__(self, path, name=None):
        self.path = path
        self.name = name + '.log' if name else '.log'
        self.stime = time.time()
        self.etime = None
        self.log_file = open(os.path.join(self.path, self.name), 'w')

    def write(self, **kwargs):
        """Write in log's file.

        Parameters
        ----------
        kwargs: Dictionary, key = signature, value = parameter
            Arbitrary sequence of parameters.
        """

        for a in kwargs:
            self.log_file.write(str(a.upper()) + ": ")
            self.log_file.write(str(kwargs[a]) + '\n')
    
    def close(self):
        """Closes the log's file.
        """

        self.etime = time.time()
        a = self.etime-self.stime
        ts = '%s s = %s m' % (a, a/60)
        self.write(start_time=self.stime, end_time=self.etime, timestamp=ts)
        self.log_file.close()