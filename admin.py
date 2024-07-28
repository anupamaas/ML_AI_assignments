from django.contrib import admin
from .models import Attendance,UnrecognizedFace,Student
# Register your models here.
admin.site.register(Attendance)
admin.site.register(UnrecognizedFace)
admin.site.register(Student)