from glob import glob
import unittest
import sys
import demography

if __name__ == '__main__':
    all_tests = unittest.TestSuite()
    testfiles = glob('test_*.py')
    all_test_mods = []
    for file in testfiles:
        module = file.split('.')[0]
        mod = __import__(module)
        all_tests.addTest(mod.suite)
    unittest.TextTestRunner(verbosity=2).run(all_tests)

