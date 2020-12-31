from flask import Flask,render_template,request,redirect
import imp

app=Flask(__name__)

@app.route('/')
def function():
	result=None
	return render_template("index.html")


@app.route("/submit",methods=["POST"])
def submit():
	result=None
	di=request.files
	img=di['image']
	path="./static/{}".format(img.filename)
	img.save(path)
	seq=imp.predict(path)
	return render_template("index.html",result=seq,path=path)

if __name__=='__main__':
	app.run(debug=True)