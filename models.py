#
# from django.db import models
#
# class Attendance(models.Model):
#     student_name = models.CharField(max_length=255)
#     date = models.DateField(auto_now_add=True)
#
#     def __str__(self):
#         return self.student_name
# from django.db import models
#

from django.db import models
from django.contrib.auth.models import User
class Student(models.Model):
    name = models.CharField(max_length=255)
    image_path = models.CharField(max_length=255)

    def __str__(self):
        return self.name
class Attendance(models.Model):
    student = models.ForeignKey('Student', on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    status = models.CharField(max_length=20, default='Present')  # Add this line

    def __str__(self):
        return f"{self.student.name} - {self.date} - {self.status}"
class UnrecognizedFace(models.Model):
    image = models.ImageField(upload_to='unrecognized_faces/')
    date = models.DateField(auto_now_add=True)
