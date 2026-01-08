if (top.location == self.location)
  top.location.href = "index.html?" + document.location.href.substring(document.location.href.lastIndexOf('/')+1, document.location.href.length)
top.document.title = document.title
