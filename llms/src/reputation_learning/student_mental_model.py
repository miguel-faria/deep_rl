#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from typing import Dict, List, Union, Tuple
from student_model import StudentModel


class StudentMentalModel(StudentModel):

