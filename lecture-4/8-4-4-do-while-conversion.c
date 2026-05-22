void f() {
	while (1) {
		do_something();

		if (condition)
			break_to break_label;
	}
break_label:
}

void f() {
	do
		do_something();
	while (!condition);
}