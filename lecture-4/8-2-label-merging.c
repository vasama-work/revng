void f() {
some_loop_1:
some_loop_2:
	if (whatever) {
		do_something();

		if (whatever)
			goto some_loop_2;

		do_something();
		goto some_loop_1;
	}
}
