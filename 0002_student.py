# Generated by Django 4.2.14 on 2024-07-26 11:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('smapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Student',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('image_path', models.CharField(max_length=255)),
            ],
        ),
    ]