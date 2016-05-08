from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput({'class': 'form-control', 'placeholder': 'Email'}))
    username = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username'}))
    password1 = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Password'}))
    password2 = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirm Password'}))

    class Meta:
        model = User
        fields = {'username', 'email', 'password1', 'password2'}

    def save(self, commit = True):
        user = super(RegistrationForm, self).save(commit= False)
        user.email = self.cleaned_data['email']
        # user.username = self.cleaned_data['username']
        # user.password = self.cleaned_data['password1']
        # user.password2 = self.cleaned_data['password2']
        if commit:
            user.save()

        return user
