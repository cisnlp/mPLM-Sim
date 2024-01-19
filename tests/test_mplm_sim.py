#!/usr/bin/env python

"""Tests for `mplm_sim` package."""


import unittest

from mplm_sim import Loader
from mplm_sim import Executor

class TestMplm_sim(unittest.TestCase):
    """Tests for `mplm_sim` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_loader(self):
        """Test loader."""
        loader = Loader.from_pretrained(model_name='cis-lmu/glot500-base', corpus_name='flores200')
        sim = loader.get_sim('german', 'cmn_Hani')

    def test_executor(self):
        """Test executor."""
        # executor = Executor(model_name='cis-lmu/glot500-base', corpus_name='own', corpus_path='corpora/own', corpus_type='text')
        # executor.run()

