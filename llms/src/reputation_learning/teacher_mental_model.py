#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from typing import Dict, List, Union, Tuple
from teacher_model import TeacherModel


class TeacherMentalModel(TeacherModel):

	
